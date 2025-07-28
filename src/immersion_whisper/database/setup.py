from .models import Lemma, Subtitle, SubtitleLemma, db


def create_triggers():
    """Creates triggers for automatic frequency updates."""
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


def _setup_database_elements():
    """Shared logic for creating tables and triggers."""
    db.create_tables([Subtitle, Lemma, SubtitleLemma], safe=True)
    create_triggers()
    print("Database tables and triggers are set up.")


def init_db():
    """Safely initializes the database, creating tables only if they don't exist."""
    try:
        db.connect()
        _setup_database_elements()
    except Exception as e:
        print(f"An error occurred during DB initialization: {e}")
    finally:
        if not db.is_closed():
            db.close()


def reset_db():
    """DESTRUCTIVE: Drops all tables and re-creates the database from scratch."""
    try:
        db.connect()
        db.drop_tables([Subtitle, Lemma, SubtitleLemma], safe=True)
        _setup_database_elements()
    except Exception as e:
        print(f"An error occurred during DB reset: {e}")
    finally:
        if not db.is_closed():
            db.close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "reset":
        reset_db()
    else:
        init_db()
