from dotenv import load_dotenv

from .args import parse_args
from .core import transcriber

load_dotenv()


def main():
    args = parse_args()

    transcriber.transcribe(args.input_file)

    # translate(srt_path, args.input_file)


if __name__ == "__main__":
    main()
