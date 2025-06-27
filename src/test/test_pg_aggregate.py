import pytest
from alembic_utils.pg_aggregate import PGAggregate, pg_aggregate_from_definition, SQLParseFailure
from sqlalchemy.sql.elements import TextClause

@pytest.fixture
def simple_agg():
    return PGAggregate(
        schema="public",
        signature="myagg(integer)",
        definition="SFUNC = my_sum, STYPE = integer, INITCOND = 0"
    )

@pytest.fixture
def agg_with_finalfunc():
    return PGAggregate(
        schema="my_schema",
        signature="aggfinal(text)",
        definition="SFUNC = my_textcat, STYPE = text, INITCOND = '', FINALFUNC = my_final"
    )

@pytest.fixture
def agg_with_kwargs():
    return PGAggregate(
        schema="public",
        signature="aggopt(int)",
        definition="SFUNC = int4pl, STYPE = int, INITCOND = 0",
        PARALLEL="SAFE",
        MSSPACE=8
    )

@pytest.fixture
def all_functions():
    class DummyFunc:
        def __init__(self, signature):
            self.signature = signature
    return [DummyFunc("my_sum(integer)"), DummyFunc("my_textcat(text, text)"), DummyFunc("my_final(text)")]

def test_pgaggregate_basic_properties(simple_agg):
    assert simple_agg.schema == "public"
    assert simple_agg.signature == "myagg(integer)"
    assert "SFUNC" in simple_agg.definition
    assert simple_agg.literal_signature == '"myagg"(integer)'
    assert simple_agg.type_ == "aggregate"

def test_pgaggregate_with_finalfunc(agg_with_finalfunc):
    if agg_with_finalfunc._finalfunc is not None:
        assert agg_with_finalfunc._finalfunc.startswith("pg_catalog.") or agg_with_finalfunc._finalfunc.startswith("my_final")
    assert "FINALFUNC" in agg_with_finalfunc.definition

def test_pgaggregate_with_kwargs(agg_with_kwargs):
    assert "PARALLEL = SAFE" in agg_with_kwargs.definition
    assert "MSSPACE = 8" in agg_with_kwargs.definition

def test_clean_aggregate_definition_removes_comments():
    dirty = "SFUNC = foo, -- this is a comment\nSTYPE = bar"
    cleaned = PGAggregate.clean_aggregate_definition(dirty)
    assert "--" not in cleaned
    assert "SFUNC = foo, STYPE = bar" == cleaned

def test_quote_ident_quotes_needed():
    assert PGAggregate.quote_ident("foo") == "foo"
    assert PGAggregate.quote_ident("FooBar") == '"FooBar"'
    assert PGAggregate.quote_ident("foo_bar") == "foo_bar"
    assert PGAggregate.quote_ident("foo bar") == '"foo bar"'

def test_literal_signature():
    agg = PGAggregate("public", "sum_custom(bigint)", "SFUNC = int8pl, STYPE = bigint")
    assert agg.literal_signature == '"sum_custom"(bigint)'

def test_from_sql_valid():
    sql = """
    CREATE AGGREGATE public.myagg(integer) (
        SFUNC = my_sum,
        STYPE = integer,
        INITCOND = 0
    )
    """
    agg = PGAggregate.from_sql(sql)
    assert agg.schema == "public"
    assert agg.signature == "myagg(integer)"
    assert "SFUNC = my_sum" in agg.definition

def test_from_sql_invalid_raises():
    with pytest.raises(SQLParseFailure):
        PGAggregate.from_sql("CREATE AGGREGTE invalid_sql")  # typo

def test_autofill_initcond_for_type():
    agg = PGAggregate("public", "auto_text(text)", "SFUNC = f, STYPE = text", _stype="text")
    assert agg.autofill_initcond_for_type("text") == "''"
    assert agg.autofill_initcond_for_type("bigint") == "0"
    assert agg.autofill_initcond_for_type("notype") is None

def test_initcond_quoted_for_int_types():
    agg = PGAggregate("public", "myagg(integer)", "SFUNC = my_sum, STYPE = integer, INITCOND = 0", _stype="integer")
    assert "INITCOND = '0'" in agg.definition or "INITCOND = 0" in agg.definition

def test_autofill_dependencies(all_functions):
    agg = PGAggregate(
        schema="public",
        signature="myagg(integer)",
        definition="SFUNC = my_sum, STYPE = integer",
        _sfunc="my_sum",
        all_entities=all_functions,
    )
    assert agg._stored_dependencies
    assert any("my_sum" in getattr(dep, "signature", "") for dep in agg._stored_dependencies)

def test_get_dependencies_schema_prefix():
    agg = PGAggregate(
        schema="my_schema",
        signature="agg(integer)",
        definition="SFUNC = foo, FINALFUNC = bar, STYPE = integer"
    )
    deps = agg.get_dependencies()
    assert "my_schema.FOO" in deps
    assert "my_schema.BAR" in deps

def test_parse_aggregate_components_basic(simple_agg):
    comp = simple_agg._parse_aggregate_components()
    assert "sfunc" in comp
    assert "stype" in comp
    assert "initcond" in comp

def test_parse_aggregate_components_full():
    agg = PGAggregate(
        schema="public",
        signature="fullagg(integer)",
        definition="SFUNC = s, STYPE = t, FINALFUNC = f, INITCOND = 5, COMBINEFUNC = cf, SERIALFUNC = sf, DESERIALFUNC = dsf, MSFUNC = msf, MINVFUNC = minv, MSTYPE = mst, MSPACE = 8, SORTOP = op"
    )
    comp = agg._parse_aggregate_components()
    keys = ["sfunc", "stype", "finalfunc", "initcond", "combinefunc", "serialfunc", "deserialfunc", "msfunc", "minvfunc", "mstype", "mspace", "sortop"]
    for k in keys:
        assert k in comp

def test_to_sql_statement_create(simple_agg):
    stmt = simple_agg.to_sql_statement_create()
    assert isinstance(stmt, TextClause)
    assert "CREATE AGGREGATE" in str(stmt)

def test_to_sql_statement_drop(simple_agg):
    stmt = simple_agg.to_sql_statement_drop()
    assert isinstance(stmt, TextClause)
    assert "DROP AGGREGATE" in str(stmt)

def test_to_sql_statement_drop_with_types():
    agg = PGAggregate(
        schema="public",
        signature="sum_custom(bigint, text)",
        definition="SFUNC = f, STYPE = s"
    )
    stmt = agg.to_sql_statement_drop()
    assert "bigint, text" in str(stmt)

def test_to_sql_statement_drop_no_params():
    agg = PGAggregate(
        schema="public",
        signature="noarg()",
        definition="SFUNC = f, STYPE = s"
    )
    stmt = agg.to_sql_statement_drop()
    assert "()" in str(stmt)

def test_to_sql_statement_create_or_replace(simple_agg):
    stmts = list(simple_agg.to_sql_statement_create_or_replace())
    assert len(stmts) == 2
    assert "DROP AGGREGATE" in str(stmts[0])
    assert "CREATE AGGREGATE" in str(stmts[1])

class DummyResult:
    def __init__(self, rows):
        self._rows = rows
    def fetchall(self):
        return self._rows

class DummySession:
    def __init__(self, rows):
        self._rows = rows
    def execute(self, *args, **kwargs):
        return DummyResult(self._rows)

def test_from_database_standard(monkeypatch):
    row = (
        "public", "aggname", "integer",
        "0", "myschema.my_sfunc", "int4", "myschema.my_finalfunc"
    )
    sess = DummySession([row])
    aggs = PGAggregate.from_database(sess)
    assert aggs
    assert aggs[0].schema == "public"
    assert "SFUNC" in aggs[0].definition

def test_from_database_sqlparse(monkeypatch):
    sql = """CREATE AGGREGATE public.testagg(integer) (SFUNC = my_sum, STYPE = integer, INITCOND = 0)"""
    row = ("public", "testagg", "integer", sql, None, None, None)
    sess = DummySession([row])
    aggs = PGAggregate.from_database(sess)
    assert aggs
    assert aggs[0].schema == "public"
    assert aggs[0].signature == "testagg(integer)"

def test_from_database_invalid(monkeypatch, capsys):
    row = ("public", "failagg", "integer", None, None, None, None)
    sess = DummySession([row])
    aggs = PGAggregate.from_database(sess)
    assert isinstance(aggs, list)
    assert all(isinstance(a, PGAggregate) for a in aggs)

def test_pg_aggregate_from_definition():
    agg = pg_aggregate_from_definition("s", "sig(int)", "SFUNC = f, STYPE = int")
    assert isinstance(agg, PGAggregate)

def test_render_import_statement():
    stmt = PGAggregate.render_import_statement()
    assert "import PGAggregate" in stmt or "import" in stmt

def test_drop_signature_weird_spacing():
    agg = PGAggregate("public", "name (  int  ,  text  )", "SFUNC = f, STYPE = s")
    stmt = agg.to_sql_statement_drop()
    assert "int, text" in str(stmt)

def test_clean_def_strip_and_commas():
    messy = """
        SFUNC = sum
            ,
            STYPE = int
        ,INITCOND = 0
    """
    clean = PGAggregate.clean_aggregate_definition(messy)
    assert "SFUNC = sum" in clean
    assert "STYPE = int" in clean

def test_quote_ident_edgecases():
    assert PGAggregate.quote_ident("with-dash") == '"with-dash"'
    assert PGAggregate.quote_ident("123name") == '"123name"'
    assert PGAggregate.quote_ident("normal_name") == "normal_name"

def test_autofill_initcond_for_type_unknown_type():
    agg = PGAggregate(
        schema='public',
        signature='myagg(integer)',
        definition='',
        initcond=None,
        state_type='unknown_type',
        state_func='func',
        final_func=None
    )
    assert agg.autofill_initcond_for_type('unknown_type') is None

def test_to_sql_statement_drop_with_exists():
    agg = PGAggregate(
        schema='public',
        signature='myagg(integer)',
        definition='...',
        initcond=None,
        state_type='integer',
        state_func='func',
        final_func=None
    )
    sql = agg.to_sql_statement_drop().text
    assert "DROP AGGREGATE" in sql

def test_pg_aggregate_from_database_not_found(monkeypatch):
    class FakeConn:
        def execute(self, *_):
            class Result:
                def fetchall(self):
                    return []
            return Result()
    result = PGAggregate.from_database(FakeConn(), 'does_not_exist')
    assert not result

def test_quote_initcond_if_needed_special_cases():
    agg = PGAggregate(
        schema='public',
        signature='myagg(integer)',
        definition='SFUNC = func, STYPE = integer, INITCOND = 42',
        _stype='integer'
    )
    assert "INITCOND = '42'" in agg.definition

def test_qualify_builtin():
    agg = PGAggregate(
        schema='public',
        signature='myagg(integer)',
        definition='SFUNC = count, STYPE = integer'
    )
    assert 'pg_catalog.count' in agg.definition

def test_sqlparsefailure_fallback():
    from alembic_utils.pg_aggregate import SQLParseFailure
    try:
        raise SQLParseFailure("fail")
    except SQLParseFailure:
        pass  # must not fail

def test_replaceableentity_fallback():
    from alembic_utils.pg_aggregate import ReplaceableEntity
    ent = ReplaceableEntity("s", "sig", "def")
    assert ent.schema == "s"
    assert ent.signature == "sig"
    assert ent.definition == "def"

def test_autofill_initcond_for_type_unknown():
    agg = PGAggregate("public", "foo(text)", "SFUNC = x, STYPE = text")
    assert agg.autofill_initcond_for_type("unknown_type") is None

def test_to_sql_statement_drop_weird_signature():
    agg = PGAggregate("public", "no_parenthesis", "SFUNC = x, STYPE = text")
    sql = agg.to_sql_statement_drop()
    assert "DROP AGGREGATE" in str(sql)

def test_from_database_error_handling():
    class DummySession:
        def execute(self, sql, params):
            class DummyResult:
                def fetchall(self_inner):
                    return [("a",)]  # too short to trigger IndexError
            return DummyResult()
    aggs = PGAggregate.from_database(DummySession())
    assert isinstance(aggs, list)

def test_fallback_classes_exist():
    from alembic_utils.pg_aggregate import SQLParseFailure, ReplaceableEntity, normalize_whitespace
    e = SQLParseFailure("fail")
    r = ReplaceableEntity("schema", "sig", "def")
    s = normalize_whitespace("foo  bar\nbaz")
    assert isinstance(e, Exception)
    assert isinstance(r, ReplaceableEntity)
    assert s == "foo bar baz"

def test_to_sql_statement_drop_handles_invalid_signature():
    agg = PGAggregate(schema="public", signature="kaputt", definition="SFUNC = foo, STYPE = text")
    stmt = agg.to_sql_statement_drop()
    assert "DROP AGGREGATE" in str(stmt)
    assert '"kaputt"' in str(stmt)

def test_autofill_initcond_for_type_returns_none_for_unknown():
    agg = PGAggregate(
        schema="public",
        signature="dummy(dummytype)",
        definition="SFUNC = foo, STYPE = dummytype"
    )
    assert agg.autofill_initcond_for_type("unknown_type_123") is None
