import peewee as pw

db = pw.SqliteDatabase("subtitles.db")


class BaseModel(pw.Model):
    class Meta:
        database = db


class Subtitle(BaseModel):
    id = pw.AutoField()
    text = pw.TextField()
    episode_number = pw.IntegerField()
    starts_at = pw.FloatField()
    ends_at = pw.FloatField()


class Lemma(BaseModel):
    id = pw.AutoField()
    text = pw.CharField(unique=True)
    frequency = pw.IntegerField(default=0)
    definition = pw.TextField(null=True)

    card_subtitle = pw.ForeignKeyField(Subtitle, null=True)


class SubtitleLemma(BaseModel):
    id = pw.AutoField()
    subtitle = pw.ForeignKeyField(Subtitle, backref="lemmas")
    lemma = pw.ForeignKeyField(Lemma, backref="subtitles")
