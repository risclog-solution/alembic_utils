from parse import parse
from sqlalchemy import text as sql_text

from alembic_utils.exceptions import SQLParseFailure
from alembic_utils.on_entity_mixin import OnEntityMixin
from alembic_utils.replaceable_entity import ReplaceableEntity
from alembic_utils.statement import coerce_to_quoted


class PGPolicy(OnEntityMixin, ReplaceableEntity):
    """A PostgreSQL Policy compatible with `alembic revision --autogenerate`

    **Parameters:**

    * **schema** - *str*: A SQL schema name
    * **signature** - *str*: A SQL policy name and tablename, separated by "."
    * **definition** - *str*:  The definition of the policy, incl. permissive, for, to, using, with check
    * **on_entity** - *str*:  fully qualifed entity that the policy applies
    """

    type_ = "policy"

    def __init__(self, *args, enable_rls=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_rls = enable_rls

    @classmethod
    def from_sql(cls, sql: str) -> "PGPolicy":
        """Create an instance instance from a SQL string"""

        template = "create policy{:s}{signature}{:s}on{:s}{on_entity}{:s}{definition}"
        result = parse(template, sql.strip(), case_sensitive=False)

        if result is not None:

            on_entity = result["on_entity"]
            if "." not in on_entity:
                schema = "public"
                on_entity = schema + "." + on_entity
            schema, _, _ = on_entity.partition(".")

            return cls(  # type: ignore
                schema=schema,
                signature=result["signature"],
                definition=result["definition"],
                on_entity=on_entity,
            )
        raise SQLParseFailure(f'Failed to parse SQL into PGPolicy """{sql}"""')

    def to_sql_statement_create(self):
        sqls = []
        if getattr(self, "enable_rls", False):
            sqls.append(f"ALTER TABLE {self.on_entity} ENABLE ROW LEVEL SECURITY;")
        sqls.append(f"CREATE POLICY {self.signature} on {self.on_entity} {self.definition}")

        return sql_text("\n".join(sqls))

    def to_sql_statement_drop(self, cascade=False):
        cascade_clause = "cascade" if cascade else ""
        sqls = [f"DROP POLICY {self.signature} on {self.on_entity} {cascade_clause};"]
        if getattr(self, "enable_rls", False):
            sqls.append(f"ALTER TABLE {self.on_entity} DISABLE ROW LEVEL SECURITY;")
        return sql_text("\n".join(sqls))

    def to_sql_statement_create_or_replace(self):
        """Not implemented, postgres policies do not support replace."""
        yield sql_text(f"DROP POLICY IF EXISTS {self.signature} on {self.on_entity};")
        yield sql_text(f"CREATE POLICY {self.signature} on {self.on_entity} {self.definition};")

    @classmethod
    def from_database(cls, connection, schema):
        """Get a list of all policies defined in the db"""
        sql = sql_text(
            f"""
        select
            schemaname,
            tablename,
            policyname,
            permissive,
            roles,
            cmd,
            qual,
            with_check
        from
            pg_policies
        where
            schemaname = '{schema}'
        """
        )
        rows = connection.execute(sql).fetchall()

        def get_definition(permissive, roles, cmd, qual, with_check):
            definition = ""
            if permissive is not None:
                definition += f"as {permissive} "
            if cmd is not None:
                definition += f"for {cmd} "
            if roles is not None:
                definition += f"to {', '.join(roles)} "
            if qual is not None:
                if qual[0] != "(":
                    qual = f"({qual})"
                definition += f"using {qual} "
            if with_check is not None:
                if with_check[0] != "(":
                    with_check = f"({with_check})"
                definition += f"with check {with_check} "
            return definition

        db_policies = []
        for schema, table, policy_name, permissive, roles, cmd, qual, with_check in rows:
            definition = get_definition(permissive, roles, cmd, qual, with_check)

            schema = coerce_to_quoted(schema)
            table = coerce_to_quoted(table)
            policy_name = coerce_to_quoted(policy_name)
            policy = cls.from_sql(f"create policy {policy_name} on {schema}.{table} {definition}")
            db_policies.append(policy)

        for policy in db_policies:
            assert policy is not None

        return db_policies
