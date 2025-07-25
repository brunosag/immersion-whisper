import peewee as pw

db = pw.SqliteDatabase("subtitles.db")


class BaseModel(pw.Model):
    class Meta:
        database = db


class Subtitle(BaseModel):
    id = pw.AutoField()
    text = pw.TextField()
    episode_number = pw.IntegerField()
    starts_at = pw.IntegerField()
    ends_at = pw.IntegerField()


class Lemma(BaseModel):
    id = pw.AutoField()
    text = pw.CharField(unique=True)
    frequency = pw.IntegerField(default=0)
    definition = pw.TextField()

    card_subtitle = pw.ForeignKeyField(Subtitle, null=True)


class SubtitleLemma(BaseModel):
    subtitle = pw.ForeignKeyField(Subtitle, backref="lemmas")
    lemma = pw.ForeignKeyField(Lemma, backref="subtitles")

    class Meta:  # type: ignore
        primary_key = pw.CompositeKey("subtitle", "lemma")


def create_tables():
    try:
        db.connect()
        db.create_tables([Subtitle, Lemma, SubtitleLemma], safe=True)
    except Exception as e:
        print(f"Error while creating tables: {e}")
    finally:
        if not db.is_closed():
            db.close()
