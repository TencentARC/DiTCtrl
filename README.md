# DiTCtrl: Exploring Attention Control in Multi-Modal Diffusion Transformer for Tuning-Free Multi-Prompt Longer Video Generation



**[Minghong Cai<sup>1 &dagger;</sup>](https://onevfall.github.io/personal_page/), 
[Xiaodong Cun<sup>2</sup>](https://vinthony.github.io/academic/), 
[Xiaoyu Li<sup>3 &#9993;</sup>](https://xiaoyu258.github.io/), 
[Wenze Liu<sup>1</sup>](https://openreview.net/profile?id=~Wenze_Liu1), 
[Zhaoyang Zhang<sup>3</sup>](https://zzyfd.github.io/#/), 
[Yong Zhang<sup>4</sup>](https://yzhang2016.github.io/), 
[Ying Shan<sup>3</sup>](https://www.linkedin.com/in/YingShanProfile/), 
[Xiangyu Yue<sup>1 &#9993;</sup>](https://xyue.io/)**
<br>
<sup>1</sup>MMLab, The Chinese University of Hong Kong
<sup>2</sup>GVC Lab, Great Bay University
<sup>3</sup>ARC Lab, Tencent PCG
<sup>4</sup>Tencent AI Lab
<br>
&dagger;: Intern at ARC Lab, Tencent PCG, &#9993;: Corresponding Authors

<a href='https://arxiv.org/abs/2412.18597'><img src='https://img.shields.io/badge/ArXiv-2412.18597-red'></a> 
<a href='https://onevfall.github.io/project_page/ditctrl/'><img src='https://img.shields.io/badge/Project-Page-Green'></a>


<div >
    <img src="assets/teaser.gif" >
</div>

## 📋 News
- [2024.12.24] Release code and demo on CogVideoX-2B!
- [2025.1.3] Release code of DiT attention map visualization.

## 🔆Demo

### Longer Multi-Prompt Text-to-video Generation

![mp_demo](assets/mp_oneline.gif)

<br>

### Longer Single-Prompt Text-to-video Generation
Our method can naturally work on single-prompt longer video generation by setting sequential multi-prompts as the same. This shows that our method can enhance the consistency of single prompt in long video generation.

![sp_demo](assets/sp_oneline.gif)

<br>

### Video Editing
Removing our latent blending strategy of our approach DiTCtrl,
we can achieve the video editing performance of **Word Swap** like [prompt-to-prompt](https://github.com/google/prompt-to-prompt).
Specifically, we just use KV-sharing strategy to share keys and values from source prompt P_source branch,
so that we can synthesize a new video to preserve the original composition 
while also addressing the content of the new prompt P_target.

![word_swap](assets/word_swap_2.gif)

Similar to [prompt-to-prompt](https://github.com/google/prompt-to-prompt), 
through reweighting the specific columns and rows corresponding to specified token (e.g. "pink") 
in the MM-DiT's Text-Video attention and Video-Text attention, 
we can also achieve the video editing performance of **Reweight**.

![video_reweight](assets/video_reweight_1.gif)


## 🎏 Abstract
<b>TL; DR: <font color="red">DiTCtrl</font> is the first tuning-free approach based on MM-DiT architecture for coherent multi-prompt video generation. Our key idea is to take the multi-prompt video generation task as temporal video editing with smooth transitions.</b>


<details><summary>CLICK for the full abstract</summary>


> Sora-like video generation models have achieved remarkable progress with a Multi-Modal Diffusion Transformer (MM-DiT) architecture. 
However, the current video generation models predominantly focus on single-prompt, struggling to generate coherent scenes with multiple sequential prompts that better reflect real-world dynamic scenarios. 
While some pioneering works have explored multi-prompt video generation, they face significant challenges including strict training data requirements, weak prompt following, and unnatural transitions. 
To address these problems, we propose <font color="red">DiTCtrl</font>, a training-free multi-prompt video generation method under MM-DiT architectures for the first time. 
Our key idea is to take the multi-prompt video generation task as temporal video editing with smooth transitions. 
To achieve this goal, we first analyze MM-DiT's attention mechanism, finding that the 3D full attention behaves similarly to that of the cross/self-attention blocks in the UNet-like diffusion models, enabling mask-guided precise semantic control across different prompts with attention sharing for multi-prompt video generation. 
Based on our careful design, the video generated by <font color="red">DiTCtrl</font> achieves smooth transitions and consistent object motion given multiple sequential prompts without additional training. 
Besides, we also present MPVBench, a new benchmark specially designed for multi-prompt video generation to evaluate the performance of multi-prompt generation. 
Extensive experiments demonstrate that our method achieves state-of-the-art performance without additional training.
</details>




## 🛡 Setup Environment
Our method is tested using CUDA12, on a single A100 or V100.

```bash
cd DiTCtrl

conda create -n ditctrl python=3.10
conda activate ditctrl

pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt

conda install https://anaconda.org/xformers/xformers/0.0.28.post1/download/linux-64/xformers-0.0.28.post1-py310_cu12.1.0_pyt2.4.1.tar.bz2
```

Our environment is similar to [CogVideo](https://github.com/THUDM/CogVideo/blob/main/sat/README.md). You may check them for more details.


## ⚙️ Download CogVideoX-2B Model Weights

First, download CogVideoX-2B model weights, download as follows, which is copied from [CogVideoX](https://github.com/THUDM/CogVideo/blob/main/sat/README.md): 

```
cd sat
mkdir CogVideoX-2b-sat
cd CogVideoX-2b-sat
wget https://cloud.tsinghua.edu.cn/f/fdba7608a49c463ba754/?dl=1
mv 'index.html?dl=1' vae.zip
unzip vae.zip
wget https://cloud.tsinghua.edu.cn/f/556a3e1329e74f1bac45/?dl=1
mv 'index.html?dl=1' transformer.zip
unzip transformer.zip
```

Arrange the model files in the following structure:

```
CogVideoX-2b-sat/
├── transformer
│   ├── 1000 (or 1)
│   │   └── mp_rank_00_model_states.pt
│   └── latest
└── vae
    └── 3d-vae.pt
```

Since model weight files are large, it’s recommended to use `git lfs`.  
See [here](https://github.com/git-lfs/git-lfs?tab=readme-ov-file#installing) for `git lfs` installation.

```
git lfs install
```

Next, clone the T5 model, which is used as an encoder and doesn’t require training or fine-tuning.
> You may also use the model file location on [Modelscope](https://modelscope.cn/models/ZhipuAI/CogVideoX-2b).

```
git clone https://huggingface.co/THUDM/CogVideoX-2b.git # Download model from Huggingface
# git clone https://www.modelscope.cn/ZhipuAI/CogVideoX-2b.git # Download from Modelscope
mkdir t5-v1_1-xxl
mv CogVideoX-2b/text_encoder/* CogVideoX-2b/tokenizer/* t5-v1_1-xxl
```

This will yield a safetensor format T5 file that can be loaded without error during Deepspeed fine-tuning.


```
├── added_tokens.json
├── config.json
├── model-00001-of-00002.safetensors
├── model-00002-of-00002.safetensors
├── model.safetensors.index.json
├── special_tokens_map.json
├── spiece.model
└── tokenizer_config.json

0 directories, 8 files

```

### ❓ FAQ

**Q: I'm getting a `safetensors rust.SafetensorError: Error while deserializing header: HeaderTooLarge` error. What should I do?**

**A:** It's because the T5 model not downloaded correctly. Please check the filesize of the `t5-v1_1-xxl` folder, it should be around **8.9GB**. Otherwise, you may be influenced by huggingface network. You can go to [hf-mirror](https://hf-mirror.com/) by the following command:

```
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download THUDM/CogVideoX-2b --local-dir ./CogVideoX-2b
```


Finally, your file structure should be like this:

```bash
sat/
├── CogVideoX-2b-sat/
  ├── transformer
  ├── CogVideoX-2b
  ├── t5-v1_1-xxl
  ├── vae
├── configs/
├── inference_case_configs/
├── run_multi_prompt.sh
├── run_single_prompt.sh
├── run_edit_video.sh 
├── sample_video.py
├── sample_video_edit.py
├── README.md
├── LICENSE
├── ...
```

## 💫 Get Started


### 1. Longer Multi-Prompt Text-to-Video

```bash
  cd sat
  bash run_multi_prompt.sh
```

### 2. Longer Single-Prompt Text-to-Video

```bash
  cd sat
  bash run_single_prompt.sh
```

### 3. Video Editing

```bash
  cd sat
  bash run_edit_video.sh
```

### Custom config

Take the `run_multi_prompt.sh` as an example:

```bash
inference_case_config="inference_case_configs/multi_prompts/rose.yaml"
run_cmd="$environs python sample_video.py --base configs/cogvideox_2b.yaml configs/inference.yaml --custom-config $inference_case_config"
echo ${run_cmd}
eval ${run_cmd}
```

The custom config is the config file in the `inference_case_configs` folder. `inference_case_configs` is the folder where you put your custom config files, which can **overwrite** the default config in the `configs/inference.yaml` folder.

Take the `rose.yaml` as an example:

```yaml
args:
  is_run_isolated: False  # If True, will generate the isolated videos not using our method
  seed: 42
  output_dir: outputs/multi_prompt_case/rose  # The output directory
  prompts:     # Put your prompts here to generate multi-prompt long videos
    - "A gentle close shot of the same rose petal, where the camera gradually pulls back to reveal the entire unfurling bloom in its perfect symmetry."
    - "A steady medium shot of the rose, where the camera continues retreating to show the full stem with its leaves and neighboring buds."
    - "A smooth full shot of the rose bush, where the camera moves further back to encompass the entire garden bed and surrounding flowering plants."
```
More details about the custom config, please refer to the `configs/inference.yaml` file.
When you run the command, it will generate the video in the `outputs/multi_prompt_case/rose` folder.

### How to create your own prompts by Large Language Model

**Single-prompts**: Please refer to the [CogvideoX](https://github.com/THUDM/CogVideo/blob/main/inference/convert_demo.py) instruction.

**Multi-prompts**: First, you can refer to our prompts case in the `inference_case_configs/multi_prompts` folder to get inspiration. Then, we provide two instruction files in the `prompts_gen_instruction` folder to generate your own multi-prompts. You can try both of them and chat with the LLM to get the best prompts.

- [Presto](prompts_gen_instruction/presto.md): Modified from Presto's instruction, focusing on realistic cinematographic sequences with natural camera movements and temporal progression (ideal for documentary-style or realistic scenarios).
- [DitCtrl](prompts_gen_instruction/ditctrl.md): Our custom instruction for DiTCtrl, emphasizing creative scene transitions and imaginative scenarios (perfect for artistic and fantasy-based video generation).

### How to visualize the attention maps

The code is also provided, you can run this:

```bash
  cd sat
  bash run_visualize.sh
```

## 🚧 Todo


- [x] Release paper on arxiv
- [x] Release Code based on <a href='https://github.com/THUDM/CogVideo'>CogVideoX-2B</a>
- [x] Visualization of attention maps
- [ ] Benchmark metrics
- [ ] Diffuser version of DiTCtrl on CogVideoX-2B



## 😉 Citation

```bibtex
@article{cai2024ditctrl,
  title     = {DiTCtrl: Exploring Attention Control in Multi-Modal Diffusion Transformer for Tuning-Free Multi-Prompt Longer Video Generation},
  author    = {Cai, Minghong and Cun, Xiaodong and Li, Xiaoyu and Liu, Wenze and Zhang, Zhaoyang and Zhang, Yong and Shan, Ying and Yue, Xiangyu},
  journal   = {arXiv:2412.18597},
  year      = {2024},
}
```

## 📚 Acknowledgements
Our codebase builds on [CogVideoX](https://github.com/THUDM/CogVideo), [MasaCtrl](https://github.com/TencentARC/MasaCtrl), [MimicMotion](https://github.com/Tencent/MimicMotion), [FreeNoise](https://github.com/AILab-CVC/FreeNoise), and [prompt-to-prompt](https://github.com/google/prompt-to-prompt). 
Thanks to the authors for sharing their awesome codebases! Thanks to concurrent training-based work [Presto](https://presto-video.github.io/#gallery) for providing the scene description instruction, and the first case is inspired by the scene description from [Presto](https://presto-video.github.io/#gallery). Thanks for the great work!

## License

This project is released under [LICENSE](LICENSE).
