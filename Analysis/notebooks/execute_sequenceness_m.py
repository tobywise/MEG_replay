import papermill as pm
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("session_id")
    parser.add_argument("n_runs", type=int)
    parser.add_argument("output_base", default='~/Scratch')
    parser.add_argument("--n_stim", type=int, default=7)
    parser.add_argument("--n_shift", type=int, default=-5)
    parser.add_argument("--p_shift", type=int, default=6)
    args = parser.parse_args()

    print(args.data_dir)

    pm.execute_notebook(
       r'preprocessing.ipynb',
       r'preprocessing_{0}_{1}.ipynb'.format(args.session_id, args.n_runs),
        kernel_name='python3',
       parameters = dict(data_dir=args.data_dir,
                         session_id=args.session_id,
                         n_runs=args.n_runs,
                         output_base_dir=args.output_base,
                         n_stim=args.n_stim,
                         cores=4)
    )

    pm.execute_notebook(
       r'localiser_analysis_full.ipynb',
       r'localiser_analysis_full_{0}_{1}.ipynb'.format(args.session_id, args.n_runs),
        kernel_name='python3',
       parameters = dict(data_dir=args.data_dir,
                         session_id=args.session_id,
                         output_base_dir=args.output_base,
                         n_stim=args.n_stim,
                         shifts=[args.n_shift, args.p_shift],
                         cores=4)
    )

    pm.execute_notebook(
       r'sequenceness.ipynb',
       r'sequenceness_{0}_{1}.ipynb'.format(args.session_id, args.n_runs),
        kernel_name='python3',
       parameters = dict(session_id=args.session_id,
                         output_base_dir=args.output_base,
                         n_stim=args.n_stim,
                         shifts=[args.n_shift, args.p_shift],
                         cores=4)
    )