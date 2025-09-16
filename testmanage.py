#!/usr/bin/env python

import argparse
import os
import shutil
import sys
import warnings

from django.core.management import execute_from_command_line

os.environ["DJANGO_SETTINGS_MODULE"] = "testapp.settings"
sys.path.append("tests")


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--deprecation",
        choices=["all", "pending", "imminent", "none"],
        default="imminent",
    )
    return parser


def parse_args(args=None):
    return make_parser().parse_known_args(args)


def runtests():
    args, rest = parse_args()

    if args.deprecation == "all":
        # Show all deprecation warnings from all packages
        warnings.simplefilter("default", DeprecationWarning)
        warnings.simplefilter("default", PendingDeprecationWarning)
    elif args.deprecation == "none":
        # Deprecation warnings are ignored by default
        pass

    argv = [sys.argv[0], *rest]

    execute_from_command_line(argv)


if __name__ == "__main__":
    runtests()
