import click

from a2ml.cmdl.cmdl import pass_context
from a2ml.cmdl.cmdl import pass_context
from a2ml.api import gc_a2ml
from a2ml.api import az_a2ml
import yaml

class ImportCmd(object):

    def __init__(self, ctx):
        self.ctx = ctx

    def import_data(self):
        name = self.ctx.config['name']
        providers=self.ctx.config['providers'].split(',')
        for provider in providers: 
            if (provider == "google"):
                project = self.ctx.config['google/project']
                print("Project: {}".format(project))          
                model = gc_a2ml.GCModel(name,project,self.ctx.google_config['google/region'])
                model.import_data(self.ctx.config.source)
                self.ctx.config['google'].dataset_id=model.dataset_id
                self.ctx.write_config('google') 
            elif (provider == 'azure'):
                region = self.ctx.config['azure'].get('region','eastus2')
                model = az_a2ml.AZModel(name,region)
                model.import_data(self.ctx.config.source)


@click.command('import', short_help='Import data for training.')
@pass_context
def cmdl(ctx):
    """Import data for training."""
    ctx.setup_logger(format='')
    ImportCmd(ctx).import_data()
