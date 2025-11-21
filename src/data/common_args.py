"""
Shared argument parser infrastructure for AUV data processing modules.

Provides common argument parsers to eliminate duplication across modules
and ensure consistent command-line interfaces.
"""

import argparse
from pathlib import Path

# Define constants locally to avoid circular imports
DEFAULT_BASE_PATH = Path(__file__).parent.joinpath("../../data/auv_data").resolve()
DEFAULT_FREQ = "1S"  # 1 Hz resampling frequency
DEFAULT_MF_WIDTH = 3  # Median filter width


class CommonArgumentParser:
    """Shared argument parser factory for all AUV processing modules."""

    @staticmethod
    def get_core_parser():
        """Get parser with core arguments used across all modules.

        Returns:
            argparse.ArgumentParser: Parser configured with add_help=False for parent use
        """
        parser = argparse.ArgumentParser(add_help=False)

        # Core processing arguments - used by almost all modules
        parser.add_argument(
            "--base_path",
            action="store",
            default=DEFAULT_BASE_PATH,
            help=f"Base directory for missionlogs and missionnetcdfs, default: {DEFAULT_BASE_PATH}",
        )
        parser.add_argument(
            "--auv_name",
            action="store",
            default="Dorado389",
            help="AUV name: Dorado389 (default), i2map, or multibeam",
        )
        parser.add_argument(
            "--mission",
            action="store",
            help="Mission directory, e.g.: 2020.064.10",
        )
        parser.add_argument(
            "--noinput",
            action="store_true",
            help="Execute without asking for responses, e.g. to not ask to re-download file",
        )
        parser.add_argument(
            "--verbose",
            type=int,
            choices=range(3),
            default=0,
            help="Verbosity level: 0=WARN (default), 1=INFO, 2=DEBUG",
        )

        return parser

    @staticmethod
    def get_processing_parser():
        """Get parser with common processing control arguments.

        Returns:
            argparse.ArgumentParser: Parser configured with add_help=False for parent use
        """
        parser = argparse.ArgumentParser(add_help=False)

        # Processing control arguments
        parser.add_argument(
            "--local",
            action="store_true",
            help="Specify if files are local in the MISSION directory",
        )
        parser.add_argument(
            "--clobber",
            action="store_true",
            help="Overwrite existing output files",
        )
        parser.add_argument(
            "--noreprocess",
            action="store_true",
            help="Don't re-process existing output files",
        )

        return parser

    @staticmethod
    def get_dorado_parser():
        """Get parser with Dorado-specific arguments.

        Returns:
            argparse.ArgumentParser: Parser configured with add_help=False for parent use
        """
        parser = argparse.ArgumentParser(add_help=False)

        # Dorado-specific arguments
        parser.add_argument(
            "--add_seconds",
            type=int,
            help="Add seconds for GPS Week Rollover Bug",
        )
        parser.add_argument(
            "--use_portal",
            action="store_true",
            help="Download via portal instead of mount",
        )
        parser.add_argument(
            "--freq",
            type=str,
            default=DEFAULT_FREQ,
            help=f"Resampling frequency in Hz, default: {DEFAULT_FREQ}",
        )
        parser.add_argument(
            "--mf_width",
            type=int,
            default=DEFAULT_MF_WIDTH,
            help=f"Median filter width for smoothing, default: {DEFAULT_MF_WIDTH}",
        )

        return parser

    @staticmethod
    def get_lrauv_parser():
        """Get parser with LRAUV-specific arguments.

        Returns:
            argparse.ArgumentParser: Parser configured with add_help=False for parent use
        """
        parser = argparse.ArgumentParser(add_help=False)

        # LRAUV-specific arguments
        parser.add_argument(
            "--log_file",
            action="store",
            help=(
                "Path to the log file of original LRAUV data, e.g.: "
                "brizo/missionlogs/2025/20250903_20250909/"
                "20250905T072042/202509050720_202509051653.nc4"
            ),
        )

        return parser

    @staticmethod
    def get_time_range_parser():
        """Get parser with time range filtering arguments.

        Returns:
            argparse.ArgumentParser: Parser configured with add_help=False for parent use
        """
        parser = argparse.ArgumentParser(add_help=False)

        # Time range filtering arguments
        parser.add_argument(
            "--start_year",
            type=int,
            help="Start year for mission filtering",
        )
        parser.add_argument(
            "--end_year",
            type=int,
            help="End year for mission filtering",
        )
        parser.add_argument(
            "--start_yd",
            type=int,
            help="Start year day for mission filtering",
        )
        parser.add_argument(
            "--end_yd",
            type=int,
            help="End year day for mission filtering",
        )
        parser.add_argument(
            "--last_n_days",
            type=int,
            help="Process only the last N days of data",
        )

        return parser

    @classmethod
    def create_parser(cls, module_name, parents=None, **kwargs):
        """Create a parser with standard formatting and common parents.

        Args:
            module_name: Name of the module (for help text)
            parents: List of parent parsers to include
            **kwargs: Additional arguments for ArgumentParser

        Returns:
            argparse.ArgumentParser: Configured parser
        """
        default_kwargs = {
            "formatter_class": argparse.RawTextHelpFormatter,
            "parents": parents or [],
        }
        default_kwargs.update(kwargs)

        return argparse.ArgumentParser(**default_kwargs)


# Convenience functions for common parser combinations
def get_standard_dorado_parser(**kwargs):
    """Get parser with standard Dorado arguments (core + processing + dorado)."""
    parents = [
        CommonArgumentParser.get_core_parser(),
        CommonArgumentParser.get_processing_parser(),
        CommonArgumentParser.get_dorado_parser(),
    ]
    return CommonArgumentParser.create_parser("dorado", parents=parents, **kwargs)


def get_standard_lrauv_parser(**kwargs):
    """Get parser with standard LRAUV arguments (core + processing + lrauv)."""
    parents = [
        CommonArgumentParser.get_core_parser(),
        CommonArgumentParser.get_processing_parser(),
        CommonArgumentParser.get_lrauv_parser(),
    ]
    return CommonArgumentParser.create_parser("lrauv", parents=parents, **kwargs)


def get_mission_processing_parser(**kwargs):
    """Get parser with mission processing arguments (includes time range)."""
    parents = [
        CommonArgumentParser.get_core_parser(),
        CommonArgumentParser.get_processing_parser(),
        CommonArgumentParser.get_dorado_parser(),
        CommonArgumentParser.get_time_range_parser(),
    ]
    return CommonArgumentParser.create_parser("mission_processing", parents=parents, **kwargs)
