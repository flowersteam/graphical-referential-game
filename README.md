# Contrastive Multimodal Learning for Emergence of Graphical Sensory-Motor Communication

This repository contains the code for the paper [Contrastive Multimodal Learning for Emergence of Graphical Sensory-Motor Communication](https://arxiv.org/abs/2210.06468).
It introduces a new variant of the well-known referential game, a framework used to study language emergence in populations of artificial agents.
We focus on an ecological setting where communication relies on a sensory-motor channel.
This setting is the Graphical Referential Game (GREG) where a speaker must produce a graphical utterance to name a visual referent object while a listener has to select
the corresponding object among distractor referents, given the delivered message.
To tackle the GREG this repository contains the CURVES algorithm: Contrastive Utterance-Referent associatiVE Scoring.

![Capture d’écran 2023-03-08 à 19 16 08](https://user-images.githubusercontent.com/29377658/223796457-a1c6b3a8-d1cb-4e5a-9a58-2fe9f045e0ba.png)

# Repository structure:

The repository assumes the following folders. Datasets are automatically downloaded and generated during the first experiment launch.

 📦 graphical-referential-game     
 ┣ 📂 src     
   ┣ 📂 data   
   ┣ 📂 results  
   ┣ 📜 scripts  
  
# Dependencies
Dependencies are listed in the ``dependencies.txt`` file

# Launching Experiments

> python3 main.py --exp_name "EXAMPLE" --seed 0 --max_iterations 100000

Options:
```
--exp_name		: Name of experiment
--seed			: Seed
--max_iterations	: Nb of iterations
--nb_agents		: Nb of agents
--nb_features		: Nb of basic referents in R1 (max 10 for Visual Referents!)
--assoc_lr		: Association LR
--action_lr		: Utterance Generation LR
--action_bs		: Nb of simultaneous searches per utterance generation
--action_it		: Nb of iteration per utterance generation
--action_size		: Nb of weights in the DMP (has to be even!)
--ss_n			: Nb of points in DMP trajectory
--referent_bs		: Referents batch_size (if > nb_features, does not matter)
--shared_perspective   : Wether agents should share perspectives
--no_perspective	: One-hot referents
--no_baseline		: No baseline for association updates
```
# Evaluating Experiments

> python3 eval.py --exp_name "EXAMPLE" --seed 0 

Options:
```
--exp_name		: Name of experiment
--seed			: Seed
--P                    : Nb of perspective/generation per utterance
```
