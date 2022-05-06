from pathlib import Path
from os.path import dirname
from sqlite3 import connect


def sql_make_table(conn, table_name, columns, verbose=False):
    cur = conn.cursor()
    cur.execute("SELECT count(name) FROM sqlite_master WHERE type='table' AND name='" + table_name + "';")
    if cur.fetchone()[0] != 1:
        cmd = "CREATE TABLE {} (".format(table_name)
        cmd += ", ".join(columns)
        cmd += ");"
        conn.execute(cmd)
        if verbose:
            print("Table {} was created".format(table_name))
    elif verbose:
        print("Table {} already exists".format(table_name))


def sql_drop_table(conn, table_name, verbose=True):
    cur = conn.cursor()
    cur.execute("SELECT count(name) FROM sqlite_master WHERE type='table' AND name='" + table_name + "';")
    if cur.fetchone()[0] == 1:
        cmd = "DROP TABLE {};".format(table_name)
        conn.execute(cmd)
        if verbose:
            print("Table {} was dropped".format(table_name))

    elif verbose:
        print("Table {} doesn't exist and could therefore not be dropped".format(table_name))


def sql_make_default_performance_table(conn, table_name, evaluate_train=True, evaluate_valid=True, evaluate_test=True,
                                       columns=None, verbose=False):
    if columns is None:
        columns = []
    columns = ["dataset TINYTEXT",
               "nr_hidden TINYTEXT",
               "readout_function TINYTEXT",
               "loss_function TINYTEXT",
               "learning_rate FLOAT",
               "early_stopping BOOL",
               "nr_epochs SMALLINT",
               "batch_size SMALLINT",
               "seed SMALLINT"] + columns
    if evaluate_train:
        columns += ["train_performance FLOAT", "train_loss FLOAT"]
    if evaluate_valid:
        columns += ["validation_performance FLOAT", "validation_loss FLOAT"]
    if evaluate_test:
        columns += ["test_performance FLOAT", "test_loss FLOAT"]
    sql_make_table(conn, table_name, columns, verbose=verbose)


def sql_connect(db_name):
    Path(dirname(db_name)).mkdir(parents=True, exist_ok=True)
    conn = connect(db_name)
    return conn


def sql_disconnect(conn):
    conn.close()


def sql_add_performance(conn, table, result):
    columns = ','.join(result.keys())
    placeholders = ', '.join('?' * len(result.values()))
    cmd = 'INSERT INTO {}({}) VALUES({})'.format(table, columns, placeholders)
    print(cmd, result.values())
    conn.execute(cmd, tuple(result.values()))


def sql_upsert_performance(conn, table, result, unique_cols=None):

    # Check if row already exists
    if unique_cols is None:
        unique_cols = ["dataset", "nr_hidden", "readout_function", "loss_function", "learning_rate",
                       "early_stopping", "nr_epochs", "batch_size", "seed"]
    cur = conn.cursor()
    where_cmd = ' WHERE (' + ' = ? AND '.join(unique_cols) + ' = ?)'
    cmd = "SELECT count(*) FROM {}{};".format(table, where_cmd)
    cur.execute(cmd, tuple(result[k] for k in unique_cols))

    # Either update or add performance
    if cur.fetchone()[0] == 1:
        performance_keys = ["train_performance", "train_loss", "validation_performance",
                            "validation_loss", "test_performance", "test_loss"]
        set_cmd = ""
        perfs = []
        for pkey in performance_keys:
            if pkey in result.keys():
                set_cmd += " {} = ?,".format(pkey)
                perfs += [result[pkey]]
            print(set_cmd)
        cmd = "UPDATE {} SET{}{};".format(table, set_cmd[:-1], where_cmd)
        print(cmd)
        conn.execute(cmd, tuple(perfs + [result[k] for k in unique_cols]))
    else:
        sql_add_performance(conn, table, result)


def sql_delete(conn, table, result, unique_cols=None):
    if unique_cols is None:
        unique_cols = ["dataset", "nr_hidden", "readout_function", "loss_function", "learning_rate",
                       "early_stopping", "nr_epochs", "batch_size", "seed"]
    cmd = "DELETE FROM " + table + " WHERE (" + ' = ? AND '.join(unique_cols) + ' = ?);'
    conn.execute(cmd, tuple(result[k] for k in unique_cols))


def sql_replace_performance(conn, table, result, unique_cols=None):
    sql_delete(conn, table, result, unique_cols)
    sql_add_performance(conn, table, result)


def sql_get(conn, query_dict):
    cur = conn.cursor()
    cur.execute("SELECT * FROM Posts")
    print(cur.fetchall())
    raise NotImplementedError
