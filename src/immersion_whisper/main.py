from dotenv import load_dotenv

from .args import parse_args
from .core.transcriber import transcribe
from .core.translator import translate
from .database.setup import reset_db

load_dotenv()


def main():
    args = parse_args()
    reset_db()
    srt_path = transcribe(args.input_file)
    translate(srt_path, args.input_file)


if __name__ == '__main__':
    main()
