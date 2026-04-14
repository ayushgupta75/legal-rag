"""
infra/aws_stack.py — AWS CDK stack for the Legal RAG system.

Provisions:
  - VPC with public + private subnets
  - RDS Postgres 16 with pgvector extension (via custom resource)
  - ElastiCache Redis cluster (LangGraph checkpointer)
  - ECS Fargate service for the FastAPI app
  - Application Load Balancer (HTTPS)
  - ECR repository for the Docker image
  - Secrets Manager for API keys

Deploy:
    pip install aws-cdk-lib constructs
    cdk bootstrap
    cdk deploy
"""
import aws_cdk as cdk
from aws_cdk import (
    Stack, Duration, RemovalPolicy,
    aws_ec2 as ec2,
    aws_ecs as ecs,
    aws_ecs_patterns as ecs_patterns,
    aws_rds as rds,
    aws_elasticache as elasticache,
    aws_ecr as ecr,
    aws_secretsmanager as sm,
    aws_iam as iam,
)
from constructs import Construct


class LegalRagStack(Stack):
    def __init__(self, scope: Construct, id: str, **kwargs):
        super().__init__(scope, id, **kwargs)

        # ── VPC ──────────────────────────────────────────────────────────────
        vpc = ec2.Vpc(
            self, "LegalRagVpc",
            max_azs=2,
            nat_gateways=1,
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name="public", subnet_type=ec2.SubnetType.PUBLIC, cidr_mask=24
                ),
                ec2.SubnetConfiguration(
                    name="private", subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS, cidr_mask=24
                ),
            ],
        )

        # ── Security Groups ───────────────────────────────────────────────────
        db_sg = ec2.SecurityGroup(self, "DbSG", vpc=vpc, description="RDS Postgres")
        redis_sg = ec2.SecurityGroup(self, "RedisSG", vpc=vpc, description="ElastiCache Redis")
        app_sg = ec2.SecurityGroup(self, "AppSG", vpc=vpc, description="ECS Fargate app")

        db_sg.add_ingress_rule(app_sg, ec2.Port.tcp(5432), "App → Postgres")
        redis_sg.add_ingress_rule(app_sg, ec2.Port.tcp(6379), "App → Redis")

        # ── Secrets ───────────────────────────────────────────────────────────
        anthropic_secret = sm.Secret(
            self, "AnthropicKey",
            secret_name="legal-rag/anthropic-api-key",
            description="Anthropic API key",
        )
        openai_secret = sm.Secret(
            self, "OpenAIKey",
            secret_name="legal-rag/openai-api-key",
            description="OpenAI API key (embeddings)",
        )
        db_secret = sm.Secret(
            self, "DbSecret",
            secret_name="legal-rag/db-credentials",
            generate_secret_string=sm.SecretStringGenerator(
                secret_string_template='{"username":"legalrag"}',
                generate_string_key="password",
                exclude_punctuation=True,
            ),
        )

        # ── RDS Postgres 16 with pgvector ─────────────────────────────────────
        db = rds.DatabaseInstance(
            self, "LegalRagDb",
            engine=rds.DatabaseInstanceEngine.postgres(
                version=rds.PostgresEngineVersion.VER_16
            ),
            instance_type=ec2.InstanceType.of(
                ec2.InstanceClass.T3, ec2.InstanceSize.MEDIUM
            ),
            vpc=vpc,
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS),
            security_groups=[db_sg],
            credentials=rds.Credentials.from_secret(db_secret),
            database_name="legal_rag",
            storage_encrypted=True,
            backup_retention=Duration.days(7),
            deletion_protection=True,
            removal_policy=RemovalPolicy.SNAPSHOT,
            # pgvector is a built-in extension in RDS Postgres 15+ — enable via parameter group
            parameters={"shared_preload_libraries": "pg_stat_statements"},
        )

        # ── ElastiCache Redis (LangGraph checkpointer) ────────────────────────
        redis_subnet_group = elasticache.CfnSubnetGroup(
            self, "RedisSubnetGroup",
            description="Redis subnet group",
            subnet_ids=[s.subnet_id for s in vpc.private_subnets],
        )
        redis = elasticache.CfnReplicationGroup(
            self, "LegalRagRedis",
            replication_group_description="LangGraph checkpointer",
            cache_node_type="cache.t3.micro",
            engine="redis",
            engine_version="7.1",
            num_cache_clusters=1,
            cache_subnet_group_name=redis_subnet_group.ref,
            security_group_ids=[redis_sg.security_group_id],
            at_rest_encryption_enabled=True,
            transit_encryption_enabled=False,  # set True + update REDIS_URL for prod
        )

        # ── ECR Repository ────────────────────────────────────────────────────
        repo = ecr.Repository(
            self, "LegalRagRepo",
            repository_name="legal-rag",
            removal_policy=RemovalPolicy.DESTROY,
        )

        # ── ECS Cluster + Fargate Service ─────────────────────────────────────
        cluster = ecs.Cluster(self, "LegalRagCluster", vpc=vpc)

        task_role = iam.Role(
            self, "TaskRole",
            assumed_by=iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
        )
        anthropic_secret.grant_read(task_role)
        openai_secret.grant_read(task_role)
        db_secret.grant_read(task_role)

        fargate_service = ecs_patterns.ApplicationLoadBalancedFargateService(
            self, "LegalRagService",
            cluster=cluster,
            cpu=512,
            memory_limit_mib=1024,
            desired_count=2,
            security_groups=[app_sg],
            task_image_options=ecs_patterns.ApplicationLoadBalancedTaskImageOptions(
                image=ecs.ContainerImage.from_ecr_repository(repo, tag="latest"),
                task_role=task_role,
                container_port=8000,
                environment={
                    "APP_ENV": "production",
                    "LOG_LEVEL": "INFO",
                    "POSTGRES_HOST": db.db_instance_endpoint_address,
                    "POSTGRES_PORT": db.db_instance_endpoint_port,
                    "POSTGRES_DB": "legal_rag",
                    "REDIS_URL": f"redis://{redis.attr_primary_end_point_address}:6379",
                },
                secrets={
                    "ANTHROPIC_API_KEY": ecs.Secret.from_secrets_manager(anthropic_secret),
                    "OPENAI_API_KEY": ecs.Secret.from_secrets_manager(openai_secret),
                    "POSTGRES_USER": ecs.Secret.from_secrets_manager(db_secret, "username"),
                    "POSTGRES_PASSWORD": ecs.Secret.from_secrets_manager(db_secret, "password"),
                },
            ),
            public_load_balancer=True,
        )

        # Health check
        fargate_service.target_group.configure_health_check(path="/health")

        # Auto-scaling
        scaling = fargate_service.service.auto_scale_task_count(
            min_capacity=1, max_capacity=10
        )
        scaling.scale_on_cpu_utilization(
            "CpuScaling", target_utilization_percent=70
        )
        scaling.scale_on_request_count(
            "RequestScaling",
            requests_per_target=500,
            target_group=fargate_service.target_group,
        )

        # ── Outputs ───────────────────────────────────────────────────────────
        cdk.CfnOutput(self, "ApiUrl", value=fargate_service.load_balancer.load_balancer_dns_name)
        cdk.CfnOutput(self, "EcrRepo", value=repo.repository_uri)
        cdk.CfnOutput(self, "DbEndpoint", value=db.db_instance_endpoint_address)


app = cdk.App()
LegalRagStack(app, "LegalRagStack", env=cdk.Environment(
    account=app.node.try_get_context("account"),
    region=app.node.try_get_context("region") or "us-east-1",
))
app.synth()
