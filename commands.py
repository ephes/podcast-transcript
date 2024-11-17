import fnmatch
import os
import argparse

from pathlib import Path


def get_project_root():
    return Path(__file__).parent.resolve()


def llm_content():
    """
    Output all relevant code / documentation in the project including
    the relative path and content of each file.
    """

    def echo_filename_and_content(files):
        """Print the relative path and content of each file."""
        for f in files:
            print(f)
            contents = f.read_text()
            relative_path = f.relative_to(project_root)
            print(relative_path)
            print("---")
            print(contents)
            print("---")

    project_root = Path(get_project_root())
    # Exclude files and directories. This is tuned to make the project fit into the
    # 200k token limit of the claude 3 models.
    exclude_files = {"uv.lock", "commands.py", ".gitignore", ".python-version"}
    exclude_dirs = {
        ".git",
        ".venv",
        ".idea",
        ".mypy_cache",
    }
    patterns = ["*.py", "*.md", "*.toml"]
    all_files = []
    for root, dirs, files in os.walk(project_root):
        root = Path(root)
        # d is the plain directory name
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for pattern in patterns:
            for filename in fnmatch.filter(files, pattern):
                if filename not in exclude_files:
                    all_files.append(root / filename)
    # print("\n".join([str(f) for f in all_files]))
    echo_filename_and_content(all_files)


def main():
    parser = argparse.ArgumentParser(
        description="Project Utility Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(
        title="Commands",
        description="Available commands",
        help="Additional help",
        dest="command",
        required=True,
    )

    # Define the 'llm_content' command
    subparsers.add_parser(
        "llm_content", help="Output all relevant code/documentation in the project"
    )
    # You can add more arguments to the llm_content command here if needed

    # Add more subcommands here if your project has other functionalities

    args = parser.parse_args()

    if args.command == "llm_content":
        llm_content()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
