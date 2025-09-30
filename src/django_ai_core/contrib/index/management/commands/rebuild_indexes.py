"""
Django management command to rebuild VectorIndexes.

This command rebuilds all registered VectorIndex instances.
"""

import logging
import time

from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from django_ai_core.contrib.index.base import registry

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Rebuild all registered VectorIndex instances"

    def add_arguments(self, parser):
        parser.add_argument(
            "index_names",
            nargs="*",
            help="Specific index names to rebuild (if not specified, rebuilds all)",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show what would be rebuilt without actually rebuilding",
        )
        parser.add_argument(
            "--verbose",
            action="store_true",
            help="Enable verbose output",
        )

    def handle(self, *args, **options):
        index_names = options.get("index_names", [])
        dry_run = options["dry_run"]
        verbose = options["verbose"]

        if verbose:
            logger.setLevel(logging.DEBUG)

        start_time = time.time()
        self.stdout.write(self.style.SUCCESS("Starting VectorIndex rebuild..."))
        self.stdout.write(f"Started at: {timezone.now()}")

        # Determine which indexes to rebuild
        if index_names:
            # Validate that all specified indexes exist
            available_indexes = registry.list()
            invalid_names = [
                name for name in index_names if name not in available_indexes
            ]
            if invalid_names:
                raise CommandError(f"Unknown index names: {invalid_names}")
            indexes_to_rebuild = index_names
        else:
            # Get all indexes
            indexes_to_rebuild = registry.list()

        if not indexes_to_rebuild:
            self.stdout.write(self.style.WARNING("No indexes registered"))
            return

        self.stdout.write(f"Found {len(indexes_to_rebuild)} index(es) to rebuild:")

        # Show what will be rebuilt
        for name in indexes_to_rebuild:
            self.stdout.write(f"  - {name}")

        if dry_run:
            self.stdout.write(
                self.style.WARNING("DRY RUN: Would rebuild the above indexes")
            )
            return

        # Rebuild indexes sequentially
        success_count, failure_count = self._rebuild_sequential(
            indexes_to_rebuild, verbose
        )

        # Show results
        elapsed_time = time.time() - start_time
        self.stdout.write("")
        self.stdout.write(self.style.SUCCESS("=== Rebuild Summary ==="))
        self.stdout.write(f"Total indexes processed: {len(indexes_to_rebuild)}")
        self.stdout.write(f"Successful rebuilds: {success_count}")

        if failure_count > 0:
            self.stdout.write(self.style.ERROR(f"Failed rebuilds: {failure_count}"))

        self.stdout.write(f"Total time: {elapsed_time:.2f} seconds")
        self.stdout.write(f"Completed at: {timezone.now()}")

        if failure_count > 0:
            raise CommandError(f"Failed to rebuild {failure_count} index(es)")

    def _rebuild_sequential(
        self, indexes_to_rebuild: list[str], verbose: bool
    ) -> tuple[int, int]:
        """Rebuild indexes sequentially."""
        success_count = 0
        failure_count = 0

        for i, index_name in enumerate(indexes_to_rebuild, 1):
            try:
                start_time = time.time()
                index_cls = registry.get(index_name)

                self.stdout.write(
                    f"\n[{i}/{len(indexes_to_rebuild)}] Rebuilding index: {index_name}"
                )

                index_cls().build()
                elapsed = time.time() - start_time

                self.stdout.write(
                    self.style.SUCCESS(
                        f"  ✓ Successfully rebuilt '{index_name}' in {elapsed:.2f}s"
                    )
                )
                success_count += 1

            except Exception as e:
                self.stdout.write(
                    self.style.ERROR(f"  ✗ Failed to rebuild '{index_name}': {e}")
                )
                if verbose:
                    import traceback

                    self.stdout.write(traceback.format_exc())
                failure_count += 1

        return success_count, failure_count
