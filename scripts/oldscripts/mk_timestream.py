import argparse

from drift.core import manager
from drift.pipeline import timestream


## Read arguments in.
parser = argparse.ArgumentParser(
    description="Create the visibility timeseries corresponding to a map."
)
parser.add_argument("teldir", help="The telescope directory to use.")
parser.add_argument("outdir", help="Output directory for timeseries.")
parser.add_argument(
    "--map",
    help="Each map argument is a map which contributes to the timeseries.",
    action="append",
)
parser.add_argument(
    "--noise",
    help="Number of days of co-added data (affects noise amplitude).",
    metavar="NDAYS",
    default=None,
    type=int,
)
parser.add_argument(
    "--resolution",
    help="Approximate time resolution in seconds.",
    metavar="NSEC",
    default=0,
    type=float,
)
args = parser.parse_args()

m = manager.ProductManager.from_config(args.teldir)

timestream.simulate(
    m, args.outdir, args.map, ndays=args.noise, resolution=args.resolution
)
