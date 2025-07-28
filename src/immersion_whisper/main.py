from dotenv import load_dotenv

from .args import parse_args
from .core import transcriber
from .database.setup import reset_db

load_dotenv()


def main():
    args = parse_args()
    reset_db()
    transcriber.transcribe(args.input_file)
    # translator.translate(srt_path, args.input_file)


if __name__ == "__main__":
    main()
