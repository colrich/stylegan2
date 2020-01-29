# Copyright (c) 2019, NVIDIA Corporation. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://nvlabs.github.io/stylegan2/license.html

import json
import argparse
import numpy as np
import dnnlib
import dnnlib.tflib as tflib
import re
import sys
import os
import csv

import projector
import pretrained_networks
from training import dataset
from training import misc

#----------------------------------------------------------------------------

def project_image(proj, targets, png_prefix, num_snapshots):
    snapshot_steps = set(proj.num_steps - np.linspace(0, proj.num_steps, num_snapshots, endpoint=False, dtype=int))
    misc.save_image_grid(targets, png_prefix + 'target.png', drange=[-1,1])
    proj.start(targets)
    while proj.get_cur_step() < proj.num_steps:
        print('\r%d / %d ... ' % (proj.get_cur_step(), proj.num_steps), end='', flush=True)
        proj.step()
        if proj.get_cur_step() in snapshot_steps:
            misc.save_image_grid(proj.get_images(), png_prefix + 'step%04d.png' % proj.get_cur_step(), drange=[-1,1])
    print('\r%-30s\r' % '', end='', flush=True)

#----------------------------------------------------------------------------

def project_generated_images(submit_config, network_pkl, seeds, num_snapshots, truncation_psi):
    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    proj = projector.Projector()
    proj.verbose = submit_config.verbose
    proj.set_network(Gs)
    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]

    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.randomize_noise = False
    Gs_kwargs.truncation_psi = truncation_psi

    for seed_idx, seed in enumerate(seeds):
        print('Projecting seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        rnd = np.random.RandomState(seed)
        z = rnd.randn(1, *Gs.input_shape[1:])
        tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars})
        images = Gs.run(z, None, **Gs_kwargs)
        project_image(proj, targets=images, png_prefix=dnnlib.make_run_dir_path('seed%04d-' % seed), num_snapshots=num_snapshots)

#----------------------------------------------------------------------------

def project_real_images(submit_config, network_pkl, dataset_name, data_dir, num_images, num_snapshots):
    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    proj = projector.Projector()
    proj.verbose = submit_config.verbose
    proj.set_network(Gs)

    print('Loading images from "%s"...' % dataset_name)
    dataset_obj = dataset.load_dataset(data_dir=data_dir, tfrecord_dir=dataset_name, max_label_size=0, repeat=False, shuffle_mb=0)
    print('dso shape: ' + str(dataset_obj.shape) + ' vs gs shape: ' + str(Gs.output_shape[1:]))
    assert dataset_obj.shape == Gs.output_shape[1:]

    for image_idx in range(num_images):
        print('Projecting image %d/%d ...' % (image_idx, num_images))
        images, _labels = dataset_obj.get_minibatch_np(1)
        images = misc.adjust_dynamic_range(images, [0, 255], [-1, 1])
        project_image(proj, targets=images, png_prefix=dnnlib.make_run_dir_path('image%04d-' % image_idx), num_snapshots=num_snapshots)
#----------------------------------------------------------------------------

def generate_grid_of_variants(submit_config, network_pkl, truncation_psi, latents_file):
    print('starting process of generating grid of variants of ' + latents_file)

    tflib.init_tf({'rnd.np_random_seed': 1000})

    f = open(latents_file, 'r')
    original_latents = np.array(json.load(f))
    f.close()
    print('loaded original latents from ' + latents_file)

    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)

    grid_size = (32, 1)
    grid_labels = []

    grid_latents = np.ndarray(shape=(grid_size[0]*grid_size[1],512))
    for i in range(grid_size[0] * grid_size[1]):
        grid_latents[i] = original_latents

    grid_fakes = Gs.run(grid_latents, grid_labels, is_validation=True, minibatch_size=4)
    misc.save_image_grid(grid_fakes, dnnlib.make_run_dir_path('latentmod-1.png'), drange=[-1,1], grid_size=grid_size)


def get_latents_for_seeds(submit_config, network_pkl, seeds):
    print('starting process of getting latents for seeds ' + str(seeds))

    tflib.init_tf({'rnd.np_random_seed': 1000})

    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)

    for seed_idx, seed in enumerate(seeds):
        print('Projecting seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        rnd = np.random.RandomState(seed)
        z = rnd.randn(1, *Gs.input_shape[1:])
        f = open(dnnlib.make_run_dir_path(str(seed)+'.json'), 'w')
        json.dump(z.tolist(), f)
        f.close()

def find_common_latents(submit_config, network_pkl, input_dir):
    print('starting process of finding common latents in directory ' + input_dir)

    tflib.init_tf({'rnd.np_random_seed': 1000})

    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)

    # parse the seeds out of the filenames in the input directory
    seeds = []
    seed_re = re.compile('^seed(\\d+)\.png')
    for path, dirs, files in os.walk(input_dir):
        matches = [seed_re.match(fn) for fn in files]
        seeds += [match.group(1) for match in matches]
    
    # get latents for each seed in the list
    latents = {}
    print('operating on seeds: ' + str(seeds))
    for seed_idx, seed in enumerate(seeds):
        print('Projecting seed %s ...' % (seed))
        rnd = np.random.RandomState(int(seed))
        z = rnd.randn(1, *Gs.input_shape[1:])
        latents[seed] = z
#        f = open(dnnlib.make_run_dir_path(str(seed)+'.json'), 'w')
#        json.dump(z.tolist(), f)
#        f.close()

    # compute average for each latent across all seeds
    print('we have latents for ' + str(len(latents)) + ' seeds')
    sums = np.zeros(512, np.float64)
    for seed in latents:
        this_seed_latents = latents[seed]
#        print(str(this_seed_latents[0]))
        for i in range(len(this_seed_latents[0])):
            sums[i] += this_seed_latents[0][i]

    avgs = [(sums[i] / len(latents)) for i in range(len(sums))]
#    print(str(avgs))
    f = open(dnnlib.make_run_dir_path('avgs.json'), 'w')
    json.dump([avgs], f)
    f.close()


    # output the averages, and then for each seed the variance from the average for each latent
    approx0 = np.zeros(512, np.int)
    with open(dnnlib.make_run_dir_path('latents-analysis.csv'), 'w', newline='') as csvfile:
        wrtr = csv.writer(csvfile, quoting=csv.QUOTE_MINIMAL)
        wrtr.writerow(avgs)

        for seed in latents:
            this_seed_latents = latents[seed][0]
            diffs = [(this_seed_latents[i] - avgs[i]) for i in range(len(this_seed_latents))]
            wrtr.writerow(diffs)
            for i in range(len(diffs)):
                if diffs[i] <= .0000001:
                    # if the diff between this seed's latent at this position and the average at
                    # this position is approximately 0, that's a sign that this position is part of
                    # what makes this type of image appear - these are what we're trying to find
                    #print('seed ' + str(seed) + ' has approx 0 at ' + str(i))
                    approx0[i] += 1

    for i in range(len(approx0)):
        print(str(i) + ': ' + str(approx0[i]) + ' approximate zeros')

def _parse_num_range(s):
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''
    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return range(int(m.group(1)), int(m.group(2))+1)
    vals = s.split(',')
    return [int(x) for x in vals]

def _str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


#---------------------------------------------------------------------------- 30/*-1257 

_examples = '''examples:

  # Project generated images
  python %(prog)s project-generated-images --network=gdrive:networks/stylegan2-car-config-f.pkl --seeds=0,1,5

  # Project real images
  python %(prog)s project-real-images --network=gdrive:networks/stylegan2-car-config-f.pkl --dataset=car --data-dir=~/datasets

'''

#----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='''StyleGAN2 projector.

Run 'python %(prog)s <subcommand> --help' for subcommand help.''',
        epilog=_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(help='Sub-commands', dest='command')

    generate_grid_of_variants_parser = subparsers.add_parser('generate-grid-of-variants', help="Generate a grid of variants of a single generated image")
    generate_grid_of_variants_parser.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
    generate_grid_of_variants_parser.add_argument('--truncation-psi', type=float, help='Truncation psi (default: %(default)s)', default=1.0)
    generate_grid_of_variants_parser.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')
    generate_grid_of_variants_parser.add_argument('--latents-file', help='File containing a 512-element json array floats representing the latents of the image to generate variations on (default: %(default)s)', metavar='FILE', required=True)
    generate_grid_of_variants_parser.add_argument('--verbose', help='activate verbose mode during run (defaults: %(default)s)', default=False, metavar='BOOL', type=_str_to_bool)

    get_latents_for_seeds_parser = subparsers.add_parser('get-latents-for-seeds', help='Write out latents for seeds')
    get_latents_for_seeds_parser.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
    get_latents_for_seeds_parser.add_argument('--seeds', type=_parse_num_range, help='List of random seeds', default=range(3))
    get_latents_for_seeds_parser.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')
    get_latents_for_seeds_parser.add_argument('--verbose', help='activate verbose mode during run (defaults: %(default)s)', default=False, metavar='BOOL', type=_str_to_bool)

    find_common_latents_parser = subparsers.add_parser('find-common-latents', help='Write out a csv containing latents for a directory of seeds along with difference between latent and average of that latent for each item in the vector')
    find_common_latents_parser.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
    find_common_latents_parser.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')
    find_common_latents_parser.add_argument('--input-dir', help='Directory containing generated images to find common latents between (default: %(default)s)', default='latent-inputs', metavar='DIR')

    project_real_images_parser = subparsers.add_parser('project-real-images', help='Project real images')
    project_real_images_parser.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
    project_real_images_parser.add_argument('--data-dir', help='Dataset root directory', required=True)
    project_real_images_parser.add_argument('--dataset', help='Training dataset', dest='dataset_name', required=True)
    project_real_images_parser.add_argument('--num-snapshots', type=int, help='Number of snapshots (default: %(default)s)', default=5)
    project_real_images_parser.add_argument('--num-images', type=int, help='Number of images to project (default: %(default)s)', default=3)
    project_real_images_parser.add_argument('--result-dir', help='Root directory for run results (default: %(default)s)', default='results', metavar='DIR')
    project_real_images_parser.add_argument('--verbose', help='activate verbose mode during run (defaults: %(default)s)', default=False, metavar='BOOL', type=_str_to_bool)

    args = parser.parse_args()
    subcmd = args.command
    if subcmd is None:
        print ('Error: missing subcommand.  Re-run with --help for usage.')
        sys.exit(1)

    kwargs = vars(args)
    sc = dnnlib.SubmitConfig()
    sc.num_gpus = 1
    sc.submit_target = dnnlib.SubmitTarget.LOCAL
    sc.local.do_not_copy_source_files = True
    sc.run_dir_root = kwargs.pop('result_dir')
    sc.run_desc = kwargs.pop('command')
    if 'verbose' in kwargs:
        sc.verbose = kwargs.pop('verbose')
        print('setting verbose mode to ' + str(sc.verbose))

    func_name_map = {
        'generate-grid-of-variants': 'run_latentmod.generate_grid_of_variants',
        'get-latents-for-seeds': 'run_latentmod.get_latents_for_seeds',
        'find-common-latents': 'run_latentmod.find_common_latents'
    }
    dnnlib.submit_run(sc, func_name_map[subcmd], **kwargs)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
