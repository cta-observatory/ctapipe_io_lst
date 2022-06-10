import psycopg2 as pg

class LSTDatabaseConnection:

    def __init__(self, user, database, autocommit=False):
        self.autocommit = autocommit
        self.user = user
        self.database = database

    def get_calibration_data(self, table_name, run, attributes=None):
        if attributes is None:
            self.cursor.execute("SELECT * FROM %s WHERE run=%s", (table_name, run,))
        else:
            attribute_str = "%s"*len(attributes)
            self.cursor.execute(
                f"SELECT {attribute_str} FROM %s WHERE run=%s",
                (*attributes, table_name, run))
        return self.cursor.fetchone()

    def open(self):
        print(f'Connection to db {self.database!r}, user is {self.user!r}')
        self.connection = pg.connect(f'dbname={self.database} user={self.user}')
        self.cursor = self.connection.cursor
        return self

    def close(self):
        if self.connection and self.autocommit:
            self.connection.commit()
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()

