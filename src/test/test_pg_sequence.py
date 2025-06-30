import re
import pytest
from sqlalchemy import text
from sqlalchemy.sql.elements import TextClause

from alembic_utils.pg_sequence import PGSequence
from alembic_utils.replaceable_entity import register_entities
from alembic_utils.testbase import TEST_VERSIONS_ROOT, run_alembic_command


@pytest.fixture(autouse=True)
def ensure_public_table(engine):
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE IF NOT EXISTS public.table (id integer primary key);"))
    yield

SEQ_1 = PGSequence.from_args(
    schema="public",
    signature="myseq",
    data_type="bigint",
    start_value=1,
    minvalue=1,
    maxvalue=9999,
    increment=1,
    cycle=False,
    cache=1,
    owned_by=None,
)

SEQ_2 = PGSequence.from_args(
    schema="public",
    signature="seq2",
    data_type="integer",
    start_value=10,
    minvalue=0,
    maxvalue=999,
    increment=10,
    cycle=True,
    cache=5,
    owned_by="public.table.id", 
)

ENTITIES = [SEQ_1, SEQ_2]

@pytest.fixture
def simple_seq():
    return SEQ_1

@pytest.fixture
def seq_with_owned_by():
    return SEQ_2

@pytest.fixture(autouse=True)
def clean_versions():
    for f in TEST_VERSIONS_ROOT.glob("*.py"):
        f.unlink()
    yield




def test_pgsequence_basic_properties(simple_seq):
    assert simple_seq.schema == "public"
    assert simple_seq.signature == "myseq"
    assert simple_seq.data_type == "bigint"
    assert simple_seq.start_value == 1
    assert simple_seq.increment == 1
    assert simple_seq.minvalue == 1
    assert simple_seq.maxvalue == 9999
    assert simple_seq.cache == 1
    assert simple_seq.cycle is False
    assert simple_seq.owned_by is None
    assert simple_seq.type_ == "sequence"

def test_pgsequence_with_owned_by(seq_with_owned_by):
    assert seq_with_owned_by.owned_by == "public.table.id"
    assert "OWNED BY public.table.id" in seq_with_owned_by.definition

def test_pgsequence_clean_definition_removes_comments():
    dirty = """
        -- this is a comment
        AS bigint
        START WITH 1
        INCREMENT BY 1 -- inline
        MINVALUE 1
        MAXVALUE 9999
        CACHE 1
        NO CYCLE
    """
    cleaned = PGSequence.clean_sequence_definition(dirty)
    assert "--" not in cleaned
    assert "AS bigint START WITH 1 INCREMENT BY 1 MINVALUE 1 MAXVALUE 9999 CACHE 1 NO CYCLE" == cleaned

def test_pgsequence_repr(simple_seq):
    assert "PGSequence" in repr(simple_seq)
    assert "myseq" in repr(simple_seq)

def test_render_import_statement():
    stmt = PGSequence.render_import_statement()
    assert "import PGSequence" in stmt or "import" in stmt

def test_to_sql_statement_create(simple_seq):
    stmt = simple_seq.to_sql_statement_create()
    assert isinstance(stmt, TextClause)
    assert "CREATE SEQUENCE" in str(stmt)
    assert "myseq" in str(stmt)

def test_to_sql_statement_drop(simple_seq):
    stmt = simple_seq.to_sql_statement_drop()
    assert isinstance(stmt, TextClause)
    assert "DROP SEQUENCE" in str(stmt)
    assert "myseq" in str(stmt)

def test_to_sql_statement_drop_with_cascade(simple_seq):
    stmt = simple_seq.to_sql_statement_drop(cascade=True)
    assert "CASCADE" in str(stmt)

def test_to_sql_statement_create_or_replace(simple_seq):
    stmts = list(simple_seq.to_sql_statement_create_or_replace())
    assert len(stmts) == 2
    assert "DROP SEQUENCE" in str(stmts[0])
    assert "CREATE SEQUENCE" in str(stmts[1])

def test_pgsequence_from_database(monkeypatch):
    class DummySession:
        def execute(self, sql, params=None):
            class Result:
                def mappings(self_inner):
                    class Mapper:
                        def fetchone(self_mapper):
                            return {
                                "data_type": "bigint",
                                "start_value": 1,
                                "minvalue": 1,
                                "maxvalue": 9999,
                                "increment": 1,
                                "cycle": False,
                                "cache": 1,
                            }
                    return Mapper()
            if "FROM pg_class" in str(sql):
                return type("Dummy", (), {
                    "mappings": lambda self: [{"sequence_name": "myseq", "sequence_schema": "public"}]
                })()
            return Result()
    entities = PGSequence.from_database(DummySession())
    assert isinstance(entities, list)
    assert any(isinstance(e, PGSequence) for e in entities)

def test_pgsequence_clean_def_strip_and_commas():
    messy = """
        AS bigint,
            START WITH 1,
            INCREMENT BY 1,
            MINVALUE 1,
            MAXVALUE 9999,
            CACHE 1,
            NO CYCLE
    """
    clean = PGSequence.clean_sequence_definition(messy)
    assert "AS bigint" in clean
    assert "START WITH 1" in clean
    assert "," in clean or ", " in clean


def test_create_revision(engine, ensure_public_table):
    register_entities(ENTITIES)
    run_alembic_command(
        engine=engine,
        command="revision",
        command_kwargs={"autogenerate": True, "rev_id": "seq1", "message": "create seq"},
    )

    migration_create_path = TEST_VERSIONS_ROOT / "seq1_create_seq.py"
    with migration_create_path.open() as migration_file:
        migration_contents = migration_file.read()

    assert "op.create_entity" in migration_contents
    assert "PGSequence" in migration_contents

    run_alembic_command(engine=engine, command="upgrade", command_kwargs={"revision": "head"})
    run_alembic_command(engine=engine, command="downgrade", command_kwargs={"revision": "base"})

def test_create_or_replace_no_exception():
    seq = PGSequence.from_args(
        schema="public",
        signature="myseq2",
        data_type="bigint",
        start_value=10,
        minvalue=1,
        maxvalue=1000,
        increment=1,
        cycle=False,
        cache=1,
        owned_by=None,
    )
    stmts = list(seq.to_sql_statement_create_or_replace())
    assert any("CREATE SEQUENCE" in str(s) for s in stmts)

def test_update_is_unreachable(engine, ensure_public_table):
    register_entities(ENTITIES)
    run_alembic_command(
        engine=engine,
        command="revision",
        command_kwargs={"autogenerate": True, "rev_id": "seq2", "message": "update test"},
    )
    migration_update_path = TEST_VERSIONS_ROOT / "seq2_update_test.py"
    with migration_update_path.open() as migration_file:
        migration_contents = migration_file.read()
    assert "replace_entity" not in migration_contents

def test_noop_revision(engine, ensure_public_table):
    register_entities(ENTITIES)
    run_alembic_command(
        engine=engine,
        command="revision",
        command_kwargs={"autogenerate": True, "rev_id": "noop1", "message": "initial"},
    )
    run_alembic_command(
        engine=engine,
        command="upgrade",
        command_kwargs={"revision": "head"},
    )

    register_entities(ENTITIES)
    run_alembic_command(
        engine=engine,
        command="revision",
        command_kwargs={"autogenerate": True, "rev_id": "noop2", "message": "noop"},
    )
    migration_noop_path = TEST_VERSIONS_ROOT / "noop2_noop.py"
    with migration_noop_path.open() as migration_file:
        migration_contents = migration_file.read()
    assert "create_entity" not in migration_contents
    assert "drop_entity" not in migration_contents

def reorder_drops_in_migration(contents: str) -> str:
    drop_lines = []
    for m in re.finditer(r"(.*op\\.drop_entity\\([^)]+\\).*)", contents):
        drop_lines.append(m.group(1))

    seq_lines = [l for l in drop_lines if "PGSequence" in l]
    rest_lines = [l for l in drop_lines if "PGSequence" not in l]
    ordered = seq_lines + rest_lines

    def replacer(match):
        return ordered.pop(0) if ordered else match.group(0)

    new_contents = re.sub(
        r".*op\\.drop_entity\\([^)]+\\).*", replacer, contents, count=len(drop_lines)
    )
    return new_contents

@pytest.mark.usefixtures("clean_versions")
def test_drop_sequences(engine, ensure_public_table):
    with engine.begin() as connection:
        connection.execute(SEQ_1.to_sql_statement_create())
        connection.execute(SEQ_2.to_sql_statement_create())

    register_entities([], schemas=["public"], entity_types=[PGSequence])

    run_alembic_command(
        engine=engine,
        command="revision",
        command_kwargs={"autogenerate": True, "rev_id": "drop_seq", "message": "drop all"},
    )

    migration_path = TEST_VERSIONS_ROOT / "drop_seq_drop_all.py"
    with migration_path.open() as migration_file:
        migration_contents = migration_file.read()
    migration_contents = reorder_drops_in_migration(migration_contents)

    assert "op.drop_entity" in migration_contents
    assert "PGSequence" in migration_contents

    drop_lines = [
        (m.group(), i)
        for i, m in enumerate(re.finditer(r"op\\.drop_entity\\(([^)]+)\\)", migration_contents))
    ]

    for line, _ in drop_lines:
        assert "PGSequence" in line

    run_alembic_command(engine=engine, command="upgrade", command_kwargs={"revision": "head"})
    run_alembic_command(engine=engine, command="downgrade", command_kwargs={"revision": "base"})


@pytest.mark.usefixtures("engine")
def test_pgsequence_from_database_pg_sequence(engine):
    seq_name = "test_seq_pg10"
    with engine.begin() as conn:
        conn.execute(text(f"""
            CREATE SEQUENCE public.{seq_name}
                AS integer
                START WITH 11
                INCREMENT BY 5
                MINVALUE 3
                MAXVALUE 222
                CACHE 4
                CYCLE
        """))
    try:
        with engine.connect() as sess:
            result = PGSequence.from_database(sess, schema="public")
            seq = next(s for s in result if s.signature == seq_name)
            definition = seq.definition
            assert "AS integer" in definition
            assert "START WITH 11" in definition
            assert "INCREMENT BY 5" in definition
            assert "MINVALUE 3" in definition
            assert "MAXVALUE 222" in definition
            assert "CACHE 4" in definition
            assert "CYCLE" in definition
            assert seq.schema == "public"
            assert seq.signature == seq_name
    finally:
        with engine.begin() as conn:
            conn.execute(text(f"DROP SEQUENCE IF EXISTS public.{seq_name} CASCADE"))

@pytest.mark.usefixtures("engine")
def test_pgsequence_from_database_information_schema(monkeypatch, engine):
    class DummySession:
        def execute(self, sql, params):
            sql_str = str(sql)
            if "FROM pg_class" in sql_str:
                return DummyResult([
                    {
                        "sequence_schema": "public",
                        "sequence_name": "test_seq_fallback"
                    }
                ])

            if "FROM pg_sequence" in sql_str:
                return DummyResult([])
            
            if "FROM information_schema.sequences" in sql_str:
                return DummyResult([{
                    "data_type": "bigint",
                    "start_value": 9,
                    "minvalue": 1,
                    "maxvalue": 88,
                    "increment": 2,
                    "cycle_option": "NO"
                }])
            raise AssertionError("Unbekannter SQL-Query: " + sql_str)
    class DummyResult:
        def __init__(self, rows):
            self.rows = rows
            self._i = 0
        def mappings(self):
            return self
        def __iter__(self):
            return iter(self.rows)
        def fetchone(self):
            if self._i >= len(self.rows):
                return None
            row = self.rows[self._i]
            self._i += 1
            return row

    result = PGSequence.from_database(DummySession(), schema="public")
    assert len(result) == 1
    seq = result[0]
    assert seq.schema == "public"
    assert seq.signature == "test_seq_fallback"
    definition = seq.definition
    assert "AS bigint" in definition
    assert "START WITH 9" in definition
    assert "INCREMENT BY 2" in definition
    assert "MINVALUE 1" in definition
    assert "MAXVALUE 88" in definition
    assert "CACHE 1" in definition # Fallback!
    assert "NO CYCLE" in definition

def test_pgsequence_from_database_empty(monkeypatch):
    class DummySession:
        def execute(self, sql, params):
            return DummyResult([])
    class DummyResult:
        def __init__(self, rows): self.rows = rows
        def mappings(self): return self
        def __iter__(self): return iter(self.rows)
        def fetchone(self): return None
    result = PGSequence.from_database(DummySession(), schema="public")
    assert result == []


def test_pgsequence_from_database_skip_on_missing_info(monkeypatch):
    class DummySession:
        def execute(self, sql, params):
            sql_str = str(sql)
            if "FROM pg_class" in sql_str:
                return DummyResult([
                    {
                        "sequence_schema": "public",
                        "sequence_name": "should_skip"
                    }
                ])
    
            if "FROM pg_sequence" in sql_str:
                return DummyResult([])
    
            if "FROM information_schema.sequences" in sql_str:
                return DummyResult([])
            raise AssertionError("Unerwartete Query: " + sql_str)
    class DummyResult:
        def __init__(self, rows): self.rows = rows; self._i = 0
        def mappings(self): return self
        def __iter__(self): return iter(self.rows)
        def fetchone(self):
            if self._i >= len(self.rows):
                return None
            row = self.rows[self._i]
            self._i += 1
            return row

    result = PGSequence.from_database(DummySession(), schema="public")
    assert result == []  
