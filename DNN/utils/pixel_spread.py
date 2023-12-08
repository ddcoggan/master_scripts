"""
This is an unused section of code that calculates the number of units impacted by a single voxel. I erroneously created it thinking this was the best way to measure receptive field size of units.
"""


# measure RF size for VOneNet and CORnet-S
empty = torch.zeros((1, 3, 224, 224))
dot = torch.zeros((1, 3, 224, 224))
dot[:, :, 112, 112] = 255
RF_sizes = pd.DataFrame()
for model_name in ['VOneNet', 'CORnet-S']:

    # get model
    if model_name == 'VOneNet':
        model = VOneNet(model_arch='resnet50', visual_degrees=9,gabor_seed=0)
        M = SimpleNamespace(model_name='VOneNet_resnet50',model=model)
        layer = 'vone_block'
        params = '/mnt/HDD1_12TB/repos/vonenet/vonenet/voneresnet50_e70.pth.tar'
        out_dir = f'{PROJ_DIR}/in_silico/models/VOneNet_resnet50/fMRI'
    elif model_name == 'CORnet-S':
        M = SimpleNamespace(model_name='cornet_s')
        model = get_model(M)
        layer = 'V1'
        params = '/mnt/HDD1_12TB/repos/CORnet-master/cornet/cornet_s-1d3f7974.pth'
        out_dir = f'{PROJ_DIR}/in_silico/models/cornet_s/pretrained/fMRI'

    # load model parameters
    M.model = load_params(params, model, 'model')
    M.params_loaded = True

    # get activations
    T.transform = transforms.Compose([])
    activations_empty = get_activations(M, T=T, features=empty, layers=['input',
                                                                    layer])
    activations_dot = get_activations(M, T=T, features=dot, layers=['input',
                                                                 layer])

    # check input is correct
    assert np.sum(activations_dot['input'].squeeze()) == 255*3, \
        'input appears to be more than 1 pixel'

    # use response to empty image as baseline
    RF_image = activations_dot[layer] - activations_empty[layer]

    # average across channels
    RF_image = np.mean(RF_image.squeeze(), axis=0)

    # scale back up to image size
    scale = 224 // RF_image.shape[0]
    RF_image = np.repeat(RF_image, scale, axis=1).repeat(scale, axis=0)
    assert RF_image.shape == (224, 224), 'RF image is wrong size'

    # threshold image in case of noise outside RF
    thr = 0 if model_name == 'CORnet-S' else 0.1
    RF_image = np.ceil(np.clip(RF_image-thr, 0, 1))
    RF_size_pix = np.sqrt(np.sum(RF_image))
    RF_size_deg = RF_size_pix * 9 / 224
    RF_sizes = pd.concat([RF_sizes, pd.DataFrame(
        {'model': [model_name],
         'RF_size_pix': [RF_size_pix],
         'RF_size_deg': [RF_size_deg]})])
    RF_image_RGB = np.moveaxis(np.tile(RF_image, reps=(3,1,1)), 0, -1)
    plt.imsave(f'{out_dir}/RF_size.png', RF_image_RGB)
RF_sizes.to_csv(f'{out_dir}/RF_sizes.csv')

