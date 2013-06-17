import os.path

from drift.util import config
from drift.pipeline import timestream

class PipelineManager(config.Reader):
    """Manage and run the pipeline.

    Attributes
    ----------
    timestream_directory : string
        Directory that the timestream is stored in.
    product_directory : string
        Directory that the analysis products are stored in.
    output_directory : string
        Directory to store timestream outputs in.

    generate_modes : boolean
        Calculate m-modes and svd-modes.
    generate_klmodes : boolean
        Calculate KL-modes?
    generate_powerspectra : boolean
        Estimate powerspectra?

    klmodes : list
        List of KL-filters to apply ['klname1', 'klname2', ...]
    powerspectra : list
        List of powerspectra to apply. Requires entries to be dicts
        like [ { 'psname' : 'ps1', 'klname' : 'dk'}, ...]
    """

    # Directories
    timestream_directory = config.Property(proptype=str, default='')
    product_directory = config.Property(proptype=str, default='')
    output_directory = config.Property(proptype=str, default='')

    # Actions to perform
    generate_modes = config.Property(proptype=bool, default=True)
    generate_klmodes = config.Property(proptype=bool, default=True)
    generate_powerspectra = config.Property(proptype=bool, default=True)

    # Specific products to use.
    klmodes = config.Property(proptype=list, default=[])
    powerspectra = config.Property(proptype=list, default=[])

    timestream = None
    manager = None



    def setup(self):
        """Set-up the timestream and manager objects."""

        self.timestream_directory = os.path.normpath(os.path.expandvars(os.path.expanduser(self.timestream_directory)))
        self.product_directory = os.path.normpath(os.path.expandvars(os.path.expanduser(self.product_directory)))

        self.timestream = timestream.Timestream(self.timestream_directory, self.product_directory)
        self.manager = self.timestream.manager

        if self.output_directory != '':
            self.output_directory = os.path.normpath(os.path.expandvars(os.path.expanduser(self.output_directory)))
            self.timestream.output_directory = self.output_directory


    def generate(self):
        """Generate pipeline outputs."""

        if self.generate_modes:
            print "Generating modes."

            self.timestream.generate_mmodes()
            self.timestream.generate_mmodes_svd()            

        if self.generate_klmodes:

            for klname in self.klmodes:                
                print "Generating KL filter (%s)" % klname

                self.timestream.set_kltransform(klname)
                self.timestream.generate_mmodes_kl()


        if self.generate_powerspectra:

            for ps in self.powerspectra:

                psname = ps['psname']
                klname = ps['klname']

                print "Estimating powerspectra (%s)" % psname

                self.timestream.set_kltransform(klname)
                self.timestream.set_psestimator(psname)

                self.timestream.powerspectrum()



    run = generate

