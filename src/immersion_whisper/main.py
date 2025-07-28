from dotenv import load_dotenv

from .args import parse_args
from .core import transcriber, translator

load_dotenv()


def main():
    args = parse_args()
    srt_path = transcriber.transcribe(args.input_file)
    translator.translate(srt_path, args.input_file)


if __name__ == "__main__":
    main()
