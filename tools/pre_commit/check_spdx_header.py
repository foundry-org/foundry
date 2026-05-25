# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foundry project

"""Ensure every Python source file carries the two-line SPDX header.

Used as a pre-commit hook (see `.pre-commit-config.yaml::check-spdx-header`).
Adapted from vLLM's analogous tool with the project name swapped.
"""

import sys
from enum import Enum


class SPDXStatus(Enum):
    EMPTY = "empty"  # empty __init__.py
    COMPLETE = "complete"
    MISSING_LICENSE = "missing_license"
    MISSING_COPYRIGHT = "missing_copyright"
    MISSING_BOTH = "missing_both"


FULL_SPDX_HEADER = (
    "# SPDX-License-Identifier: Apache-2.0\n"
    "# SPDX-FileCopyrightText: Copyright contributors to the Foundry project"
)

LICENSE_LINE = "# SPDX-License-Identifier: Apache-2.0"
COPYRIGHT_LINE = "# SPDX-FileCopyrightText: Copyright contributors to the Foundry project"


def check_spdx_header_status(file_path):
    with open(file_path, encoding="UTF-8") as file:
        lines = file.readlines()
        if not lines:
            return SPDXStatus.EMPTY

        start_idx = 0
        if lines[0].startswith("#!"):
            start_idx = 1

        has_license = False
        has_copyright = False

        for i in range(start_idx, len(lines)):
            line = lines[i].strip()
            if line == LICENSE_LINE:
                has_license = True
            elif line == COPYRIGHT_LINE:
                has_copyright = True

        if has_license and has_copyright:
            return SPDXStatus.COMPLETE
        if has_license:
            return SPDXStatus.MISSING_COPYRIGHT
        if has_copyright:
            return SPDXStatus.MISSING_LICENSE
        return SPDXStatus.MISSING_BOTH


def add_header(file_path, status):
    with open(file_path, "r+", encoding="UTF-8") as file:
        lines = file.readlines()
        file.seek(0, 0)
        file.truncate()

        if status == SPDXStatus.MISSING_BOTH:
            if lines and lines[0].startswith("#!"):
                file.write(lines[0])
                file.write(FULL_SPDX_HEADER + "\n")
                file.writelines(lines[1:])
            else:
                file.write(FULL_SPDX_HEADER + "\n")
                file.writelines(lines)

        elif status == SPDXStatus.MISSING_COPYRIGHT:
            for i, line in enumerate(lines):
                if line.strip() == LICENSE_LINE:
                    lines.insert(i + 1, f"{COPYRIGHT_LINE}\n")
                    break
            file.writelines(lines)

        elif status == SPDXStatus.MISSING_LICENSE:
            for i, line in enumerate(lines):
                if line.strip() == COPYRIGHT_LINE:
                    lines.insert(i, f"{LICENSE_LINE}\n")
                    break
            file.writelines(lines)


def main():
    files_missing_both = []
    files_missing_copyright = []
    files_missing_license = []

    for file_path in sys.argv[1:]:
        status = check_spdx_header_status(file_path)
        if status == SPDXStatus.MISSING_BOTH:
            files_missing_both.append(file_path)
        elif status == SPDXStatus.MISSING_COPYRIGHT:
            files_missing_copyright.append(file_path)
        elif status == SPDXStatus.MISSING_LICENSE:
            files_missing_license.append(file_path)

    all_files_to_fix = files_missing_both + files_missing_copyright + files_missing_license
    if all_files_to_fix:
        print("The following files are missing the SPDX header:")
        for file_path in files_missing_both:
            print(f"  {file_path}")
            add_header(file_path, SPDXStatus.MISSING_BOTH)
        for file_path in files_missing_copyright:
            print(f"  {file_path}")
            add_header(file_path, SPDXStatus.MISSING_COPYRIGHT)
        for file_path in files_missing_license:
            print(f"  {file_path}")
            add_header(file_path, SPDXStatus.MISSING_LICENSE)

    sys.exit(1 if all_files_to_fix else 0)


if __name__ == "__main__":
    main()
