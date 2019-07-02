# module load xorg-utils/X11R7.7
# module load freesurfer

if __name__ == "__main__":

    """
    This code runs source-space analyis on the MEG data using beamforming.

    MNE isn't set up for running source analysis in volume space easily (surface-level analysis are far more straightforward and well documented), so 
    this is a slightly awkward process. 

    MNE assumes we have MRIs for each subject, which we don't (the gain from MRI is negligible), so instead we create fake MRIs by transforming the 
    freesurfer average subject to fit our MRI data. This is done using the MNE graphical interface, lining up the MEG fiducials with the fsaverage head.

    Once we've done this, we take our new source space (based on this transformed MRI) and scale it back to the fsaverage subject and use this for further
    analysis. This means that source level analysis for every subject is carried out in fsaverage space and we can compare across subjects.

    """

    import mne
    from mne.beamformer import make_lcmv, apply_lcmv
    import os
    import nibabel as nb
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("subject")
    args = parser.parse_args()

    # Get MRI data
    subject = args.subject

    # subject = 'sub-MG05517'
    subjects_dir = '/lustre/scratch/scratch/skgttw1/replay_aversive_learning/data/derivatives/mri/'

    # Load the bem solution and source space for the subject - this has already been aligned with the MEG data
    fname_model = os.path.join(subjects_dir, subject, "bem/{0}-inner_skull-bem-sol.fif".format(subject))
    fname_src_fs  = subjects_dir + '/fsaverage/bem/fsaverage-src.fif.gz'

    # Load the fsaverage source space and scale our subject to this space so that all subjects are in fsaverage space
    if not os.path.exists(subjects_dir + '{0}/bem/{0}-src.fif.gz'.format(subject)):
        mne.scale_source_space(subject, '{subject}-src.fif.gz', subjects_dir=subjects_dir)

    # Read in the scaled source space
    src = mne.read_source_spaces(os.path.join(subjects_dir, subject, 'bem/{0}-src.fif.gz'.format(subject)))

    # Compute morphing stuff (not sure this is needed )
    morph = mne.compute_source_morph(src,
                                 subject_from=subject, subject_to='fsaverage',
                                 subjects_dir=subjects_dir)

    # Get evoked/epoched data and transformation (the file that transforms the source space to the MEG data)
    derivatives_dir = '/lustre/scratch/scratch/skgttw1/replay_aversive_learning/data/derivatives'
    evoked  = mne.read_evokeds(os.path.join(derivatives_dir, 'replay_events_evoked', '{0}_replay_events_pairwise-ave.fif.gz'.format(subject)))[0]
    epochs = mne.read_epochs(os.path.join(derivatives_dir, 'replay_events_epochs', '{0}_replay_events_pairwise-epo.fif.gz'.format(subject)))
    trans = os.path.join(derivatives_dir, 'trans', '{0}-trans.fif'.format(subject))

    # Calculate the forward solution
    fwd = mne.make_forward_solution(evoked.info, trans=trans, src=src, bem=fname_model)
    mne.write_forward_solution(os.path.join(derivatives_dir, 'forward_solutions', '{0}-fwd.fif.gz'.format(subject)), fwd, overwrite=True)

    # All epochs

    # Calculate covariance
    data_cov = mne.compute_covariance(epochs, method='shrunk', rank=None, tmin=-0)

    # Do beamforming
    filters = make_lcmv(evoked.info, fwd, data_cov, reg=0.05, pick_ori='max-power', weight_norm='nai', rank=None)
    stc = apply_lcmv(evoked, filters, max_ori_out='signed')
    stc.save(os.path.join(derivatives_dir, 'source_estimates', 'replay', '{0}'.format(subject)))
    nifti = morph.apply(stc, output='nifti1')
    nb.save(nifti, os.path.join(derivatives_dir, 'source_estimates', 'replay', '{0}.nii.gz'.format(subject)))


    # Chosen
    evoked  = mne.read_evokeds(os.path.join(derivatives_dir, 'replay_events_evoked', '{0}_replay_events_chosen-ave.fif.gz'.format(subject)))[0]
    epochs = mne.read_epochs(os.path.join(derivatives_dir, 'replay_events_epochs', '{0}_replay_events_chosen-epo.fif.gz'.format(subject)))

    # Calculate covariance
    data_cov = mne.compute_covariance(epochs, method='shrunk', rank=None, tmin=0)

    # Do beamforming
    filters = make_lcmv(evoked.info, fwd, data_cov, reg=0.05, pick_ori='max-power', weight_norm='nai', rank=None)
    stc = apply_lcmv(evoked, filters, max_ori_out='signed')
    stc.save(os.path.join(derivatives_dir, 'source_estimates', 'chosen', '{0}'.format(subject)))
    nifti = morph.apply(stc, output='nifti1')
    nb.save(nifti, os.path.join(derivatives_dir, 'source_estimates', 'chosen', '{0}.nii.gz'.format(subject)))


    # Unchosen

    evoked  = mne.read_evokeds(os.path.join(derivatives_dir, 'replay_events_evoked', '{0}_replay_events_unchosen-ave.fif.gz'.format(subject)))[0]
    epochs = mne.read_epochs(os.path.join(derivatives_dir, 'replay_events_epochs', '{0}_replay_events_unchosen-epo.fif.gz'.format(subject)))

    # Calculate covariance
    data_cov = mne.compute_covariance(epochs, method='shrunk', rank=None, tmin=0)

    # Do beamforming
    filters = make_lcmv(evoked.info, fwd, data_cov, reg=0.05, pick_ori='max-power', weight_norm='nai', rank=None)
    stc = apply_lcmv(evoked, filters, max_ori_out='signed')
    stc.save(os.path.join(derivatives_dir, 'source_estimates', 'unchosen', '{0}'.format(subject)))
    nifti = morph.apply(stc, output='nifti1')
    nb.save(nifti, os.path.join(derivatives_dir, 'source_estimates', 'unchosen', '{0}.nii.gz'.format(subject)))


    # Reactivations

    evoked  = mne.read_evokeds(os.path.join(derivatives_dir, 'replay_events_evoked', '{0}_reactivation_events-ave.fif.gz'.format(subject)))[0]
    epochs = mne.read_epochs(os.path.join(derivatives_dir, 'replay_events_epochs', '{0}_reactivation_events-epo.fif.gz'.format(subject)))

    # Calculate covariance
    data_cov = mne.compute_covariance(epochs, method='shrunk', rank=None, tmin=0)

    # Do beamforming
    filters = make_lcmv(evoked.info, fwd, data_cov, reg=0.05, pick_ori='max-power', weight_norm='nai', rank=None)
    stc = apply_lcmv(evoked, filters, max_ori_out='signed')
    stc.save(os.path.join(derivatives_dir, 'source_estimates', 'reactivation', '{0}'.format(subject)))
    nifti = morph.apply(stc, output='nifti1')
    nb.save(nifti, os.path.join(derivatives_dir, 'source_estimates', 'reactivation', '{0}.nii.gz'.format(subject)))
