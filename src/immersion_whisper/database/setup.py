from .models import Lemma, Subtitle, SubtitleLemma, db


def create_triggers():
    increment_trigger = """
    CREATE TRIGGER IF NOT EXISTS increment_frequency
    AFTER INSERT ON subtitlelemma
    FOR EACH ROW
    BEGIN
        UPDATE lemma SET frequency = frequency + 1 WHERE id = NEW.lemma_id;
    END;
    """

    decrease_trigger = """
    CREATE TRIGGER IF NOT EXISTS decrease_frequency
    AFTER DELETE ON subtitlelemma
    FOR EACH ROW
    BEGIN
        UPDATE lemma SET frequency = frequency - 1 WHERE id = OLD.lemma_id;
    END;
    """

    db.execute_sql(increment_trigger)
    db.execute_sql(decrease_trigger)


def create_db():
    try:
        db.connect()
        db.drop_tables([Subtitle, Lemma, SubtitleLemma])
        db.create_tables([Subtitle, Lemma, SubtitleLemma])
        create_triggers()
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if not db.is_closed():
            db.close()
