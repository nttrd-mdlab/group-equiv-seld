# Group Equivariant DNN for Ambisonic Signal Processing

This repository is the demonstration of the group equivariant Ambisonic signal processing DNNs [1], implemented by the authors.

## License

This repository (except submodules) is released under the specific license. **Read [License](https://github.com/nttrd-mdlab/group-equiv-seld/blob/main/LICENSE) file in this repository before you download and use this software.**

The submodule `seld-dcase2019` is under its own license.

The script `fCGModule.py` is from [zlin7/CGNet](https://github.com/zlin7/CGNet), which is originally released under the MIT License.

## Contents

```
.
├── LICENSE
├── README.md
├── adversarial_attack.py
├── article_figure
│   └── taslp
├── boot_tensorboard.sh
├── checkpoints
├── dcase19_dataset.py
├── docker
│   ├── Dockerfile
│   └── build.sh
├── evaluation.py
├── fCGModule.py
├── feature_extraction.py
├── login_torch_sh.sh
├── main.py
├── math_util.py
├── models.py
├── modules.py
├── parameter.py
├── render_taslp_fig3.py
├── render_taslp_fig4.py
├── result
├── ret_adv
├── ret_eval
├── run_adversarial_attack.sh
├── run_experiment.sh
└── seld-dcase2019
```

## Usage

We assume the environment that `docker/Dockerfile` appropriately works.

1. Clone this repository.

```sh
git clone --recursive https://github.com/nttrd-mdlab/group-equiv-seld
cd group-equiv-seld
```

2. Build the Docker environment.

```sh
$ cd docker
$ ./build.sh
> ...
> Successfully built 31cc484c9976
> Successfully tagged cgdcase:0.2
$ cd ../
```

3. Download the dataset files from the link on [this website](http://dcase.community/challenge2019/task-sound-event-localization-and-detection). You need `foa_dev.z**`, `metadata_dev.zip`, `foa_eval.zip`, `metadata_eval.zip`. Then, generate the normalized dataset using `feature_extraction.py` (do not forget to rewrite the path to the downloaded files in `feature_extraction.py`).

```sh
./login_torch_sh.sh
python3  feature_extraction.py
exit
```

4. Start model training.

```
./run_experiment.sh 0  # Specify the GPU number (0-origin) by argument
```

Trained model is saved to `./checkpoints`, and the log is saved to `./result`.

5. Change experiment conditions by rewriting `parameter.py` and re-run `./run_experiment.sh`:
   - Toggle `model=['Conventional', 'Proposed'][1]` to `[0]` to test baseline model.
   - Toggle `scale_equivariance=True` to `False` to disable scale equivariance of proposed method.
   - Switch `train_rotation_bias=['virtual_rot', 'azi_random', None][0]` to `[1]` to enable rotational data augmentation.
   - Rewrite `feature_phase_different_bin=0` to `None` to disable time translation invariance of proposed method.

6. Check and compare performance.

Evaluate the trained model
```sh
$ ./login_torch_sh.sh 0
$ python3 evaluation.py --resume ./checkpoints/(name of checkpoint file).checkpoint
$ exit
```

Compare the progress of (being) trained models
```sh
$ ./boot_tensorboard.sh
```
Then, view `http://localhost:6006` with your browser.

7. Render the figures on the paper:

```sh
$ ./login_torch_sh.sh 0
$ python3 render_taslp_fig3.py
$ python3 render_taslp_fig4.py
$ exit
```

8. Run experiment for adversarial attack.

```sh
$ ./run_adversarial_attack.sh 0 ./checkpoints/(name of checkpoint file).checkpoint (output file name)
```

## References

- [1] R. Sato, K. Niwa, K. Kobayashi, "Ambisonic Signal Processing DNNs Guaranteeing Rotation, Scale and Time Translation Equivariance," IEEE/ACM Trans. ASLP, (to be published), 2021.
