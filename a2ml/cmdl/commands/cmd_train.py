import click

from a2ml.cmdl.cmdl import pass_context
from a2ml.api import gc_a2ml
from a2ml.api import az_a2ml
import yaml
import os

class TrainCmd(object):

    def __init__(self, ctx):
        self.ctx = ctx

    def train(self):
        providers=self.ctx.config.providers.split(',')
        for provider in providers: 
            if provider == "google":
                model = gc_a2ml.GCModel(self.ctx.config.name,
                    self.ctx.config['google'].project,
                    self.ctx.config.region)
                model.train(self.ctx.config['google'].dataset_id,
                    self.ctx.config.target,
                    self.ctx.config.exclude.split(','),
                    self.ctx.config.budget,
                    self.ctx.config.get('metric','MINIMIZE_MAE'))
            elif provider == "azure":
                # name,project_id,compute_region,compute_name
                model = az_a2ml.AZModel(self.ctx.config.name,
                    self.ctx.config.region)
                model.train(self.ctx.config.source,
                    self.ctx.config.target,
                    self.ctx.config.get('exclude','').split(','),
                    self.ctx.config.get('budget',3600),
                    self.ctx.config.get('metric','spearman_correlation'))

@click.command('train', short_help='Train the model.')
@pass_context
def cmdl(ctx):
    """Train the model."""
    ctx.setup_logger(format='')
    TrainCmd(ctx).train()
