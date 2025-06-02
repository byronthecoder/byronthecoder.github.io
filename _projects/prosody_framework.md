---
layout: page
title: Prosody Analysis Framework
description: Dynamic prosodic analysis using deep learning for speech interaction research
img: assets/img/prosody_analysis.jpg
importance: 2
category: research
github: https://github.com/byronthecoder/prosody-framework
related_publications: yuan2024prosody
published: false
---

## Project Overview

The Prosody Analysis Framework is a comprehensive toolkit for analyzing prosodic features in speech interaction. This project, developed in collaboration with Dr. Leonardo Lancia at LPL (Aix-Marseille University), explores prosody as a dynamic coordinative device in human communication.

## Core Features

### Dynamic Prosodic Analysis

- **Real-time Prosody Extraction**: Advanced algorithms for extracting fundamental frequency, intensity, and timing patterns
- **Syllable-level Analysis**: Detailed examination of prosodic structures at syllable boundaries
- **Word-level Coordination**: Investigation of prosodic coordination across word boundaries

### Deep Learning Integration

- **Neural Prosody Models**: Custom architectures for prosodic pattern recognition
- **Temporal Dynamics**: Modeling of prosodic changes over conversational time
- **Cross-speaker Analysis**: Comparative prosodic analysis between conversation partners

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/prosody_features.png" title="Prosodic Feature Extraction" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/prosody_neural.png" title="Neural Network Architecture" class="img-fluid rounded z-depth-1" %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/prosody_dynamics.png" title="Dynamic Analysis Results" class="img-fluid rounded z-depth-1" %}
    </div>
</div>
<div class="caption">
    Prosodic analysis pipeline: feature extraction (left), neural network processing (center), and dynamic coordination analysis (right).
</div>

## Technical Stack

```python
# Example of prosodic feature extraction
import prosody_framework as pf

# Load audio and extract features
audio_data = pf.load_audio("conversation.wav")
prosodic_features = pf.extract_prosody(
    audio_data,
    features=['f0', 'intensity', 'duration'],
    window_size=0.025,
    hop_length=0.010
)

# Apply deep learning model
model = pf.ProsodyNet(input_dim=prosodic_features.shape[1])
coordination_scores = model.predict(prosodic_features)
```

### Key Technologies

- **Python**: Core development language
- **TensorFlow/Keras**: Deep learning implementation
- **Parselmouth**: Praat integration for Python
- **Matplotlib/Seaborn**: Visualization
- **Jupyter**: Interactive analysis notebooks

## Research Applications

### Speech Interaction Mechanisms

Understanding how speakers coordinate prosodic patterns during natural conversation, revealing insights into:

- Turn-taking behaviors
- Emotional synchronization
- Social bonding through speech

### Clinical Applications

Potential applications in:

- Speech therapy assessment
- Communication disorder diagnosis
- Social skill training programs

## Current Research

This ongoing project at **Laboratoire Parole et Langage (LPL)** focuses on:

- Neural mechanisms of prosodic coordination
- Cross-linguistic prosodic patterns
- Real-time prosodic feedback systems

{% bibliography --cited %}

## Collaboration

This project is part of the research initiative "**Prosody AS Dynamic COordinative Device**" in collaboration with:

- Dr. Leonardo Lancia (LPL, Aix-Marseille University)
- Laboratoire Parole et Langage research team
- International speech processing community
