import papermill as pm
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("session_id")
    parser.add_argument("blink_components")
    parser.add_argument("--n_runs", type=int, default=9)
    parser.add_argument("--output_dir", default='data/derivatives')
    parser.add_argument("--n_stim", type=int, default=8)
    parser.add_argument("--n_shift", type=int, default=-5)
    parser.add_argument("--p_shift", type=int, default=6)
    args = parser.parse_args()

    if args.blink_components == 'None':
        args.blink_components = None
    else:
        args.blink_components = [int(i) for i in args.blink_components.split(',')]

    # pm.execute_notebook(
    #    r'notebooks/preprocessing_template.ipynb',
    #    r'notebooks/preprocessing/sub-{0}_preprocessing.ipynb'.format(args.session_id),
    #     kernel_name='meg', start_timeout=1240,
    #    parameters = dict(data_dir=args.data_dir,
    #                      session_id=args.session_id,
    #                      n_runs=args.n_runs,
    #                      output_dir=args.output_dir,
    #                      n_stim=args.n_stim,
    #                      cores=1,
    #                      blink_components=args.blink_components)
    # )

    # pm.execute_notebook(
    #     r'notebooks/localiser_template.ipynb',
    #     r'notebooks/localiser/sub-{0}_localiser.ipynb'.format(args.session_id),
    #     kernel_name='meg',start_timeout=240,
    #     parameters = dict(data_dir=args.data_dir,
    #                      session_id=args.session_id,
    #                      output_dir=args.output_dir,
    #                      n_stim=args.n_stim,
    #                      shifts=[args.n_shift, args.p_shift],
    #                      cores=1)
    # )

    pm.execute_notebook(
        r'notebooks/sequenceness_template.ipynb',
        r'notebooks/sequenceness/sub-{0}_sequenceness.ipynb'.format(args.session_id),
        kernel_name='meg', start_timeout=240,
        parameters = dict(session_id=args.session_id,
                         output_dir=args.output_dir,
                         n_stim=args.n_stim,
                         shifts=[args.n_shift, args.p_shift],
                         cores=1)
    )

    # pm.execute_notebook(
    #    r'notebooks/outcome_decoding_template.ipynb',
    #    r'notebooks/outcome_decoding/sub-{0}_outcome_decoding.ipynb'.format(args.session_id),
    #     kernel_name='meg', start_timeout=240,
    #    parameters = dict(data_dir=args.data_dir,
    #                      session_id=args.session_id,
    #                      output_dir=args.output_dir,
    #                      n_stim=args.n_stim,
    #                      shifts=[args.n_shift, args.p_shift],
    #                      cores=1,
    #                      start_timeout=120)
    # )