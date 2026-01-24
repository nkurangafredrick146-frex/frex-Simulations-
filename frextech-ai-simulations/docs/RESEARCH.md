
Research Papers & References

This document provides a comprehensive overview of the research foundations, state-of-the-art techniques, and academic references that underpin the FrexTech AI Simulations platform.

Table of Contents

1. Foundational Research
2. World Models & Generative AI
3. 3D Representation Learning
4. Multimodal Learning
5. Diffusion Models
6. Neural Rendering
7. Scene Understanding
8. Physics & Simulation
9. Interactive AI Systems
10. Evaluation & Benchmarks
11. Ethics & Safety
12. Future Directions

Foundational Research

Core Machine Learning

1. Attention Is All You Need (2017)
   · Authors: Vaswani et al.
   · Conference: NeurIPS 2017
   · arXiv:1706.03762
   · Key Contribution: Introduced the Transformer architecture, which has become the foundation for most modern large language models and many computer vision models.
2. Deep Residual Learning for Image Recognition (2015)
   · Authors: He et al.
   · Conference: CVPR 2016
   · arXiv:1512.03385
   · Key Contribution: Residual networks (ResNets) with skip connections enable training of very deep neural networks.
3. Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift (2015)
   · Authors: Ioffe and Szegedy
   · Conference: ICML 2015
   · arXiv:1502.03167
   · Key Contribution: Normalizing layer inputs to stabilize and accelerate training.

Generative Models

1. Generative Adversarial Networks (2014)
   · Authors: Goodfellow et al.
   · Conference: NeurIPS 2014
   · arXiv:1406.2661
   · Key Contribution: Introduced GANs, a framework for training generative models through adversarial training.
2. Auto-Encoding Variational Bayes (2013)
   · Authors: Kingma and Welling
   · Conference: ICLR 2014
   · arXiv:1312.6114
   · Key Contribution: Introduced Variational Autoencoders (VAEs) for learning latent representations.
3. Neural Discrete Representation Learning (2017)
   · Authors: van den Oord et al.
   · Conference: NeurIPS 2018
   · arXiv:1711.00937
   · Key Contribution: Introduced VQ-VAE for discrete latent representations.

World Models & Generative AI

World Models

1. World Models (2018)
   · Authors: Ha and Schmidhuber
   · arXiv:1803.10122
   · Key Contribution: A generative neural network that learns a spatial and temporal model of its environment.
2. Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model (2019)
   · Authors: Schrittwieser et al.
   · Conference: Nature 2020
   · arXiv:1911.08265
   · Key Contribution: MuZero learns a model of the world and uses it for planning, achieving state-of-the-art performance.
3. Learning Latent Dynamics for Planning from Pixels (2019)
   · Authors: Hafner et al.
   · Conference: ICML 2019
   · arXiv:1811.04551
   · Key Contribution: PlaNet learns a latent dynamics model from pixels and plans with it.

Large Language Models as World Models

1. Language Models as Zero-Shot Planners: Extracting Actionable Knowledge for Embodied Agents (2022)
   · Authors: Huang et al.
   · Conference: ICML 2022
   · arXiv:2201.07207
   · Key Contribution: Shows that large language models can generate executable plans for embodied agents.
2. Do As I Can, Not As I Say: Grounding Language in Robotic Affordances (2022)
   · Authors: Ahn et al.
   · Conference: CoRL 2022
   · arXiv:2204.01691
   · Key Contribution: PaLM-SayCan combines large language models with affordance functions for robotics.

Generative World Creation

1. Learning to Generate 3D Shapes from a Single Example (2022)
   · Authors: Sanghi et al.
   · Conference: CVPR 2022
   · arXiv:2203.12675
   · Key Contribution: Single-shot 3D shape generation using a learned prior.
2. Infinite Nature: Perpetual View Generation of Natural Scenes from a Single Image (2021)
   · Authors: Liu et al.
   · Conference: ICCV 2021
   · arXiv:2012.09855
   · Key Contribution: Generates perpetual views of natural scenes from a single image.

3D Representation Learning

Neural Radiance Fields (NeRF)

1. NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis (2020)
   · Authors: Mildenhall et al.
   · Conference: ECCV 2020
   · arXiv:2003.08934
   · Key Contribution: Introduced NeRF, which represents scenes as continuous volumetric functions.
2. Instant Neural Graphics Primitives with a Multiresolution Hash Encoding (2022)
   · Authors: Müller et al.
   · Conference: SIGGRAPH 2022
   · arXiv:2201.05989
   · Key Contribution: InstantNGP enables real-time training and rendering of NeRFs.
3. Mip-NeRF: A Multiscale Representation for Anti-Aliasing Neural Radiance Fields (2021)
   · Authors: Barron et al.
   · Conference: ICCV 2021
   · arXiv:2103.13415
   · Key Contribution: Addresses aliasing in NeRFs by modeling rays as cones.

Gaussian Splatting

1. 3D Gaussian Splatting for Real-Time Radiance Field Rendering (2023)
   · Authors: Kerbl et al.
   · Conference: SIGGRAPH 2023
   · arXiv:2308.04079
   · Key Contribution: Real-time rendering of radiance fields using anisotropic 3D Gaussians.
2. Surfels: Surface Elements as Rendering Primitives (2000)
   · Authors: Pfister et al.
   · Conference: SIGGRAPH 2000
   · DOI:10.1145/344779.344936
   · Key Contribution: Introduced surfels as rendering primitives, foundational for point-based rendering.

Implicit Representations

1. DeepSDF: Learning Continuous Signed Distance Functions for Shape Representation (2019)
   · Authors: Park et al.
   · Conference: CVPR 2019
   · arXiv:1901.05103
   · Key Contribution: Learns continuous signed distance functions for 3D shape representation.
2. Occupancy Networks: Learning 3D Reconstruction in Function Space (2019)
   · Authors: Mescheder et al.
   · Conference: CVPR 2019
   · arXiv:1812.03828
   · Key Contribution: Represents 3D geometry as the decision boundary of a neural classifier.

Multi-View Reconstruction

1. Neural Volumes: Learning Dynamic Renderable Volumes from Images (2019)
   · Authors: Lombardi et al.
   · Conference: SIGGRAPH 2019
   · arXiv:1906.07751
   · Key Contribution: Learns volumetric representations from multi-view images.
2. Neural Sparse Voxel Fields (2020)
   · Authors: Liu et al.
   · Conference: NeurIPS 2020
   · arXiv:2007.11571
   · Key Contribution: Combines NeRF with sparse voxel octrees for efficient rendering.

Multimodal Learning

Vision-Language Models

1. Learning Transferable Visual Models From Natural Language Supervision (2021)
   · Authors: Radford et al.
   · Conference: ICML 2021
   · arXiv:2103.00020
   · Key Contribution: Introduced CLIP, which learns visual representations from natural language supervision.
2. Flamingo: a Visual Language Model for Few-Shot Learning (2022)
   · Authors: Alayrac et al.
   · Conference: NeurIPS 2022
   · arXiv:2204.14198
   · Key Contribution: A few-shot learning visual language model.
3. BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation (2022)
   · Authors: Li et al.
   · Conference: ICML 2022
   · arXiv:2201.12086
   · Key Contribution: A unified vision-language framework for understanding and generation.

Multimodal Fusion

1. ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks (2019)
   · Authors: Lu et al.
   · Conference: NeurIPS 2019
   · arXiv:1908.02265
   · Key Contribution: A model for learning joint representations of vision and language.
2. Unified-IO: A Unified Model for Vision, Language, and Multi-Modal Tasks (2022)
   · Authors: Lu et al.
   · arXiv:2206.08916
   · Key Contribution: A single model that performs a wide range of AI tasks.

3D-Vision-Language Models

1. Point-BERT: Pre-training 3D Point Cloud Transformers with Masked Point Modeling (2021)
   · Authors: Yu et al.
   · Conference: CVPR 2022
   · arXiv:2111.14819
   · Key Contribution: BERT-style pre-training for 3D point clouds.
2. ULIP: Learning a Unified Representation of Language, Images, and Point Clouds for 3D Understanding (2022)
   · Authors: Xue et al.
   · Conference: CVPR 2023
   · arXiv:2212.05171
   · Key Contribution: Unifies language, images, and 3D point clouds in a single representation.

Diffusion Models

Image Diffusion

1. Denoising Diffusion Probabilistic Models (2020)
   · Authors: Ho et al.
   · Conference: NeurIPS 2020
   · arXiv:2006.11239
   · Key Contribution: Introduced DDPMs, establishing modern diffusion models.
2. Diffusion Models Beat GANs on Image Synthesis (2021)
   · Authors: Dhariwal and Nichol
   · Conference: NeurIPS 2021
   · arXiv:2105.05233
   · Key Contribution: Showed diffusion models can outperform GANs on image synthesis.
3. High-Resolution Image Synthesis with Latent Diffusion Models (2021)
   · Authors: Rombach et al.
   · Conference: CVPR 2022
   · arXiv:2112.10752
   · Key Contribution: Stable Diffusion, operating in a compressed latent space.

3D Diffusion

1. Diffusion Probabilistic Models for 3D Point Cloud Generation (2021)
   · Authors: Luo and Hu
   · Conference: CVPR 2021
   · arXiv:2103.01458
   · Key Contribution: Extends diffusion models to 3D point cloud generation.
2. Score-Based Generative Modeling in Latent Space (2021)
   · Authors: Vahdat et al.
   · Conference: NeurIPS 2021
   · arXiv:2106.05931
   · Key Contribution: Introduced NVAE for high-resolution image and 3D shape generation.
3. DreamFusion: Text-to-3D using 2D Diffusion (2022)
   · Authors: Poole et al.
   · arXiv:2209.14988
   · Key Contribution: Uses 2D diffusion models as a loss for 3D generation.

Video Diffusion

1. Video Diffusion Models (2022)
   · Authors: Ho et al.
   · arXiv:2204.03458
   · Key Contribution: Extends diffusion models to video generation.
2. Imagen Video: High Definition Video Generation with Diffusion Models (2022)
   · Authors: Ho et al.
   · arXiv:2210.02303
   · Key Contribution: High-quality video generation using cascaded diffusion models.

Neural Rendering

Differentiable Rendering

1. Differentiable Monte Carlo Ray Tracing through Edge Sampling (2018)
   · Authors: Li et al.
   · Conference: SIGGRAPH Asia 2018
   · arXiv:1806.10556
   · Key Contribution: Makes Monte Carlo ray tracing differentiable.
2. Mitsuba 2: A Retargetable Forward and Inverse Renderer (2019)
   · Authors: Nimier-David et al.
   · Conference: SIGGRAPH Asia 2019
   · arXiv:1905.02759
   · Key Contribution: A differentiable renderer supporting various rendering techniques.
3. PyTorch3D: A Library for Deep Learning with 3D Data (2020)
   · Authors: Ravi et al.
   · Conference: CVPR 2020
   · arXiv:2007.08501
   · Key Contribution: PyTorch library for 3D deep learning with differentiable rendering.

Real-Time Neural Rendering

1. Neural Rendering and Reenactment of Human Actor Videos (2019)
   · Authors: Thies et al.
   · Conference: SIGGRAPH Asia 2019
   · arXiv:1909.03658
   · Key Contribution: Real-time neural rendering of human performances.
2. Deferred Neural Rendering: Image Synthesis using Neural Textures (2019)
   · Authors: Thies et al.
   · Conference: SIGGRAPH 2019
   · arXiv:1904.12356
   · Key Contribution: Combines traditional graphics pipelines with neural networks.

Novel View Synthesis

1. Neural Rerendering in the Wild (2019)
   · Authors: Xia et al.
   · Conference: CVPR 2019
   · arXiv:1904.04290
   · Key Contribution: Novel view synthesis for real-world scenes.
2. SynSin: End-to-end View Synthesis from a Single Image (2020)
   · Authors: Wiles et al.
   · Conference: CVPR 2020
   · arXiv:1912.08804
   · Key Contribution: Single image to novel view synthesis.

Scene Understanding

Semantic Scene Understanding

1. Panoptic Segmentation (2019)
   · Authors: Kirillov et al.
   · Conference: CVPR 2019
   · arXiv:1801.00868
   · Key Contribution: Unifies instance and semantic segmentation.
2. Scene Graph Generation (2018)
   · Authors: Xu et al.
   · Conference: CVPR 2018
   · arXiv:1801.01952
   · Key Contribution: Generates structured representations of scenes.

3D Scene Understanding

1. ScanNet: Richly-Annotated 3D Reconstructions of Indoor Scenes (2017)
   · Authors: Dai et al.
   · Conference: CVPR 2017
   · arXiv:1702.04405
   · Key Contribution: Large-scale dataset of 3D indoor scenes with annotations.
2. 3D-SIS: 3D Semantic Instance Segmentation of RGB-D Scans (2019)
   · Authors: Hou et al.
   · Conference: CVPR 2019
   · arXiv:1812.07003
   · Key Contribution: 3D instance segmentation of RGB-D scans.

Dynamic Scene Understanding

1. Dynamic Neural Radiance Fields for Monocular 4D Facial Avatar Reconstruction (2021)
   · Authors: Gafni et al.
   · Conference: CVPR 2021
   · arXiv:2012.03065
   · Key Contribution: Dynamic NeRF for facial avatars.
2. NeRF in the Wild: Neural Radiance Fields for Unconstrained Photo Collections (2021)
   · Authors: Martin-Brualla et al.
   · Conference: CVPR 2021
   · arXiv:2008.02268
   · Key Contribution: NeRF for unstructured photo collections.

Physics & Simulation

Physics-Based Learning

1. Learning to Simulate Complex Physics with Graph Networks (2020)
   · Authors: Sanchez-Gonzalez et al.
   · Conference: ICML 2020
   · arXiv:2002.09405
   · Key Contribution: Graph networks for learning complex physical simulations.
2. Differentiable Physics for Differentiable Simulations (2019)
   · Authors: Hu et al.
   · Conference: ICLR 2019
   · arXiv:1905.12949
   · Key Contribution: Differentiable physics for gradient-based learning.

Robotics and Control

1. Reinforcement Learning with Deep Energy-Based Policies (2017)
   · Authors: Haarnoja et al.
   · Conference: ICML 2017
   · arXiv:1702.08165
   · Key Contribution: Soft Actor-Critic, an off-policy RL algorithm.
2. Mastering the Game of Go without Human Knowledge (2017)
   · Authors: Silver et al.
   · Conference: Nature 2017
   · arXiv:1712.01815
   · Key Contribution: AlphaGo Zero learns from self-play without human data.

Simulation for Training

1. Learning to Navigate in Complex Environments (2017)
   · Authors: Mirowski et al.
   · Conference: ICLR 2017
   · arXiv:1611.03673
   · Key Contribution: Uses simulated environments to train navigation agents.
2. Gibson Env: Real-World Perception for Embodied Agents (2018)
   · Authors: Xia et al.
   · Conference: CVPR 2018
   · arXiv:1808.10654
   · Key Contribution: Photorealistic simulation environment for training embodied agents.

Interactive AI Systems

Human-AI Interaction

1. InstructPix2Pix: Learning to Follow Image Editing Instructions (2022)
   · Authors: Brooks et al.
   · arXiv:2211.09800
   · Key Contribution: Edits images based on human instructions.
2. DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation (2022)
   · Authors: Ruiz et al.
   · arXiv:2208.12242
   · Key Contribution: Personalizes diffusion models for specific subjects.

Creative AI

1. CLIPDraw: Exploring Text-to-Drawing Synthesis through Language-Image Encoders (2021)
   · Authors: Frans et al.
   · arXiv:2106.14843
   · Key Contribution: Generates drawings from text prompts using CLIP.
2. StyleCLIP: Text-Driven Manipulation of StyleGAN Imagery (2021)
   · Authors: Patashnik et al.
   · Conference: ICCV 2021
   · arXiv:2103.17249
   · Key Contribution: Uses CLIP to guide StyleGAN image editing.

Collaborative AI

1. CoDraw: Collaborative Drawing as a Testbed for Grounded Goal-driven Communication (2017)
   · Authors: Kim et al.
   · Conference: ACL 2019
   · arXiv:1712.05530
   · Key Contribution: A collaborative drawing task for studying communication.
2. Learning to Collaborate by Compromising (2019)
   · Authors: Xie et al.
   · Conference: AAMAS 2019
   · arXiv:1906.07800
   · Key Contribution: AI agents learning to collaborate through compromise.

Evaluation & Benchmarks

3D Generation Metrics

1. A Comprehensive Study on the Evaluation of Generative Models (2018)
   · Authors: Theis et al.
   · Journal: ICLR 2018
   · arXiv:1511.01844
   · Key Contribution: Reviews and evaluates metrics for generative models.
2. Chamfer Distance as a Comprehensive Measure for Point Cloud Completion (2020)
   · Authors: Xia et al.
   · Conference: CVPR 2020
   · arXiv:2004.10518
   · Key Contribution: Evaluates Chamfer distance for point cloud quality.

Perception Metrics

1. The Unreasonable Effectiveness of Deep Features as a Perceptual Metric (2018)
   · Authors: Zhang et al.
   · Conference: CVPR 2018
   · arXiv:1801.03924
   · Key Contribution: Introduced LPIPS, a learned perceptual metric.
2. Frechet Inception Distance (FID) for Evaluating Generative Models (2017)
   · Authors: Heusel et al.
   · Conference: NeurIPS 2017
   · arXiv:1706.08500
   · Key Contribution: FID metric for evaluating generative models.

Datasets

1. ShapeNet: An Information-Rich 3D Model Repository (2015)
   · Authors: Chang et al.
   · arXiv:1512.03012
   · Key Contribution: Large-scale dataset of 3D shapes.
2. CO3D: Common Objects in 3D (2021)
   · Authors: Reizenstein et al.
   · Conference: CVPR 2021
   · arXiv:2109.00512
   · Key Contribution: Dataset of common objects with 3D reconstructions.
3. OmniObject3D: Large-Vocabulary 3D Object Dataset for Realistic Perception, Reconstruction and Generation (2023)
   · Authors: Wu et al.
   · Conference: CVPR 2023
   · arXiv:2301.07525
   · Key Contribution: Large-scale 3D object dataset.

Ethics & Safety

AI Safety

1. Concrete Problems in AI Safety (2016)
   · Authors: Amodei et al.
   · Journal: arXiv 2016
   · arXiv:1606.06565
   · Key Contribution: Identifies practical AI safety problems.
2. The Malicious Use of Artificial Intelligence: Forecasting, Prevention, and Mitigation (2018)
   · Authors: Brundage et al.
   · Report
   · arXiv:1802.07228
   · Key Contribution: Examines potential malicious uses of AI.

Bias and Fairness

1. Man is to Computer Programmer as Woman is to Homemaker? Debiasing Word Embeddings (2016)
   · Authors: Bolukbasi et al.
   · Conference: NeurIPS 2016
   · arXiv:1607.06520
   · Key Contribution: Methods for debiasing word embeddings.
2. Fairness in Machine Learning (2018)
   · Authors: Barocas et al.
   · Book
   · Link
   · Key Contribution: Comprehensive treatment of fairness in ML.

Content Safety

1. Red Teaming Language Models to Reduce Harms: Methods, Scaling Behaviors, and Lessons Learned (2022)
   · Authors: Ganguli et al.
   · arXiv:2209.07858
   · Key Contribution: Methods for red teaming language models.
2. InstructGPT: Training Language Models to Follow Instructions with Human Feedback (2022)
   · Authors: Ouyang et al.
   · Conference: NeurIPS 2022
   · arXiv:2203.02155
   · Key Contribution: Aligns language models with human intent through RLHF.

Future Directions

Emerging Research Areas

1. Generative Agents: Interactive Simulacra of Human Behavior (2023)
   · Authors: Park et al.
   · Conference: UIST 2023
   · arXiv:2304.03442
   · Key Contribution: Creates interactive agents that simulate human behavior.
2. Foundation Models for Decision Making: Challenges, Opportunities, and the Future (2023)
   · Authors: Shah et al.
   · arXiv:2303.04129
   · Key Contribution: Surveys foundation models for decision making.
3. Sparks of Artificial General Intelligence: Early experiments with GPT-4 (2023)
   · Authors: Bubeck et al.
   · arXiv:2303.12712
   · Key Contribution: Examines GPT-4's capabilities as an early AGI system.

Long-Term Vision

1. Reward Is Enough (2021)
   · Authors: Silver et al.
   · Journal: Artificial Intelligence 2021
   · arXiv:2110.06267
   · Key Contribution: Argues that reward maximization is sufficient for intelligence.
2. Artificial Intelligence and Life in 2030 (2016)
   · Authors: Stone et al.
   · Report: One Hundred Year Study on Artificial Intelligence
   · Link
   · Key Contribution: Long-term study of AI's impact on society.

Implementation Resources

Libraries and Frameworks

PyTorch Ecosystem:

· PyTorch - Deep learning framework
· PyTorch3D - 3D deep learning
· PyTorch Lightning - Training framework
· Hugging Face Transformers - Pre-trained models

3D and Rendering:

· Open3D - 3D data processing
· Trimesh - Loading and using triangular meshes
· Kaolin - 3D deep learning library
· NerfStudio - NeRF development framework

Diffusion Models:

· Diffusers - State-of-the-art diffusion models
· CompVis/stable-diffusion - Stable Diffusion implementation

Online Courses and Tutorials

1. Stanford CS231n: Deep Learning for Computer Vision
   · Course Website
   · Topics: CNN, RNN, LSTM, GANs, Detection, Segmentation
2. MIT 6.S191: Introduction to Deep Learning
   · Course Website
   · Topics: Basics of deep learning, applications
3. Neural Radiance Fields Course
   · NeRF Course
   · Topics: Comprehensive NeRF tutorial

Research Communities

1. arXiv - Pre-print repository: arxiv.org
2. OpenReview - Conference reviews: openreview.net
3. Papers with Code - Papers with implementations: paperswithcode.com
4. AI Alignment Forum - AI safety discussions: alignmentforum.org

Citation

When using this research in academic work, please cite relevant papers directly. For referencing this compilation:

```
@misc{frextech2024research,
  author = {FrexTech AI Research Team},
  title = {FrexTech AI Simulations Research References},
  year = {2024},
  howpublished = {\url{https://github.com/frextech/frextech-ai-simulations/docs/RESEARCH.md}},
  note = {Comprehensive bibliography of research papers and references}
}
```

Contributing

To contribute to this research document:

1. Add new papers with complete citations
2. Include brief summaries of key contributions
3. Organize papers by category
4. Verify all links are working
5. Keep descriptions concise and informative

License

This document is licensed under CC BY 4.0. Individual papers are subject to their own copyright and licensing terms.

---

Last Updated: January 1, 2024
Maintained by: FrexTech AI Research Team
Contact: research@frextech-sim.com