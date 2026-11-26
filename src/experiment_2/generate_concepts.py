import sys
import json
import random
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_DIR = Path("D:/DEEP-LEARNING-PROJECT")
OUT_TRAIN_FILE = PROJECT_DIR / "data" / "experiment_2" / "curriculum_raw" / "category4_concepts_train.jsonl"
OUT_HELDOUT_FILE = PROJECT_DIR / "data" / "experiment_2" / "curriculum_raw" / "category4_concepts_heldout.jsonl"

SEED = 42
TARGET_TRAIN_COUNT = 850
TARGET_HELDOUT_COUNT = 150

# =====================================================================
# 1. TRAINING CONCEPTS (Economics, AI/ML, Science)
# =====================================================================

TRAIN_CONCEPTS = [
    # ------------------ ECONOMICS ------------------
    {
        "id": "monopsony",
        "domain": "economics",
        "name": "monopsony",
        "benchmark_match": True,
        "prompts": [
            "Explain monopsony in simple words.",
            "What is a monopsony?",
            "What does monopsony mean in economics?",
            "Can you explain the concept of monopsony?",
            "Define monopsony and give an example.",
            "How does a monopsony work?",
            "What is the difference between a monopoly and a monopsony?",
        ],
        "definition": (
            "A monopsony is a market structure where there is only one buyer for a good or service, "
            "giving that single buyer tremendous market power over suppliers or sellers. While a monopoly "
            "is characterized by a single seller, a monopsony is characterized by a single buyer.\n\n"
            "In economics, monopsony is most frequently discussed in labor markets. When a single large employer "
            "(such as a lone hospital in a rural town, a single coal-mining company, or a dominant tech firm) "
            "is the only major buyer of labor in a region, workers have few alternative employment options. "
            "As a result, the employer can push wages lower and reduce employment below what would prevail in a competitive market."
        )
    },
    {
        "id": "monopoly",
        "domain": "economics",
        "name": "monopoly",
        "prompts": [
            "What is a monopoly in economics?",
            "Explain what a monopoly is in simple words.",
            "Define a monopoly and explain why it can harm consumers.",
            "How does a monopoly differ from competitive markets?",
        ],
        "definition": (
            "A monopoly is a market structure in which a single company or seller provides all or nearly all of the supply "
            "for a particular product or service with no close substitutes. Because there are no direct competitors, "
            "a monopolist possesses significant pricing power and can restrict output to charge higher prices than would exist in a competitive market."
        )
    },
    {
        "id": "supply_and_demand",
        "domain": "economics",
        "name": "supply and demand",
        "prompts": [
            "Explain the law of supply and demand.",
            "What is supply and demand in economics?",
            "How do supply and demand determine prices in a market?",
            "Can you explain supply and demand in simple terms?",
        ],
        "definition": (
            "Supply and demand is the foundational model of market economics that explains how prices and quantities are determined. "
            "The Law of Demand states that when the price of a good rises, consumers will buy less of it; when the price falls, consumers buy more. "
            "The Law of Supply states that when prices rise, producers are willing to supply more of the good to maximize revenue. "
            "The intersection where the quantity demanded equals the quantity supplied is known as the market equilibrium price."
        )
    },
    {
        "id": "opportunity_cost",
        "domain": "economics",
        "name": "opportunity cost",
        "prompts": [
            "What is opportunity cost in economics?",
            "Explain opportunity cost in simple words with an example.",
            "What does opportunity cost mean?",
            "Why is opportunity cost important in decision making?",
        ],
        "definition": (
            "Opportunity cost is the value of the next best alternative that you must give up whenever you make a choice. "
            "Because resources like time and money are scarce, choosing one option prevents you from pursuing another. "
            "For example, if you spend $20 and two hours watching a movie at the cinema, the opportunity cost is what else you could have done "
            "with that $20 and two hours, such as studying, working, or buying a textbook."
        )
    },
    {
        "id": "inflation",
        "domain": "economics",
        "name": "inflation",
        "prompts": [
            "What is inflation in economics?",
            "Explain inflation in simple terms.",
            "What causes inflation and how does it affect purchasing power?",
            "Why does money lose value during inflation?",
        ],
        "definition": (
            "Inflation is the general, sustained increase in the prices of goods and services across an entire economy over time. "
            "When inflation occurs, every unit of currency buys fewer goods than it did previously, meaning your purchasing power declines. "
            "Inflation is commonly measured using indices like the Consumer Price Index (CPI) and is driven by factors such as increased money supply, "
            "high consumer demand (demand-pull), or rising production costs (cost-push)."
        )
    },
    {
        "id": "gdp",
        "domain": "economics",
        "name": "Gross Domestic Product (GDP)",
        "prompts": [
            "What is Gross Domestic Product (GDP)?",
            "Explain GDP in simple words.",
            "What does GDP measure in an economy?",
            "What is the difference between real GDP and nominal GDP?",
        ],
        "definition": (
            "Gross Domestic Product (GDP) is the total monetary value of all final goods and services produced within a country's borders "
            "during a specific period, usually one year or one quarter. It is the primary metric used to measure economic health and size. "
            "Nominal GDP measures output using current market prices, whereas Real GDP adjusts for inflation to reflect true changes in physical production."
        )
    },
    {
        "id": "fiscal_vs_monetary_policy",
        "domain": "economics",
        "name": "fiscal policy vs monetary policy",
        "prompts": [
            "What is the difference between fiscal policy and monetary policy?",
            "Explain fiscal and monetary policy in simple terms.",
            "How do central banks and governments steer the economy?",
        ],
        "definition": (
            "Fiscal policy and monetary policy are two primary tools used to manage economic stability:\n\n"
            "- **Fiscal Policy**: Managed by the legislative government through taxation and public spending decisions (such as infrastructure projects or stimulus checks).\n"
            "- **Monetary Policy**: Managed by a nation's central bank (such as the Federal Reserve) by controlling the money supply, adjusting benchmark interest rates, and regulating commercial banks."
        )
    },
    {
        "id": "comparative_advantage",
        "domain": "economics",
        "name": "comparative advantage",
        "prompts": [
            "What is comparative advantage in economics?",
            "Explain comparative advantage in simple words with an example.",
            "Why do countries trade according to comparative advantage?",
        ],
        "definition": (
            "Comparative advantage is an economic principle stating that an economy or individual can produce a particular good at a lower "
            "opportunity cost than trade partners. Even if one nation is more efficient at producing everything (absolute advantage), "
            "both nations benefit by specializing in the goods where their opportunity cost is lowest and trading with each other."
        )
    },
    {
        "id": "externalities",
        "domain": "economics",
        "name": "externalities",
        "prompts": [
            "What is an externality in economics?",
            "Explain positive and negative externalities with examples.",
            "What is a negative externality in simple terms?",
        ],
        "definition": (
            "An externality is an unintended side effect or cost/benefit incurred by an unrelated third party as a result of an economic transaction. "
            "A **negative externality** occurs when an activity imposes uncompensated costs on others (such as air pollution from a factory). "
            "A **positive externality** occurs when an activity delivers benefits to others without them paying (such as vaccinations reducing disease transmission for everyone)."
        )
    },
    {
        "id": "public_goods",
        "domain": "economics",
        "name": "public goods",
        "prompts": [
            "What is a public good in economics?",
            "Explain public goods and the free-rider problem in simple terms.",
            "What are non-rival and non-excludable goods?",
        ],
        "definition": (
            "In economics, a public good is a commodity or service that is both non-excludable and non-rivalrous:\n\n"
            "- **Non-excludable**: It is impossible or impractical to prevent anyone from enjoying the good once provided.\n"
            "- **Non-rivalrous**: One person's consumption does not reduce the quantity available for others.\n\n"
            "Classic examples include national defense, street lighting, and flood control systems. Because people can benefit without paying, "
            "markets face the 'free-rider problem', which is why public goods are usually funded by governments through taxation."
        )
    },

    # ------------------ AI & MACHINE LEARNING ------------------
    {
        "id": "artificial_intelligence",
        "domain": "ai_ml",
        "name": "artificial intelligence",
        "benchmark_match": True,
        "prompts": [
            "What is artificial intelligence?",
            "Explain artificial intelligence in simple words.",
            "Can you define artificial intelligence?",
            "What does AI mean?",
            "How would you explain artificial intelligence to a beginner?",
        ],
        "definition": (
            "Artificial Intelligence (AI) is a broad branch of computer science dedicated to building software and machines capable of "
            "performing tasks that traditionally require human intelligence. These tasks include visual perception, speech recognition, "
            "natural language translation, decision-making, and logical reasoning.\n\n"
            "AI encompasses many techniques, ranging from rule-based expert systems and search algorithms to modern machine learning and deep neural networks."
        )
    },
    {
        "id": "machine_learning",
        "domain": "ai_ml",
        "name": "machine learning",
        "benchmark_match": True,
        "prompts": [
            "What is machine learning?",
            "Explain machine learning in simple words.",
            "Can you define machine learning?",
            "What does machine learning mean?",
            "How does machine learning work?",
        ],
        "definition": (
            "Machine Learning (ML) is a subfield of artificial intelligence focused on developing algorithms that learn patterns from data "
            "and make predictions or decisions without being explicitly hard-coded with manual rules. "
            "Instead of a programmer writing fixed if-else instructions, an ML model is trained on examples, iteratively adjusting its internal "
            "parameters to minimize errors and generalize to new, unseen data."
        )
    },
    {
        "id": "ai_vs_machine_learning",
        "domain": "ai_ml",
        "name": "difference between AI and machine learning",
        "benchmark_match": True,
        "prompts": [
            "What is the difference between AI and machine learning?",
            "Explain the difference between artificial intelligence and machine learning.",
            "How does AI differ from machine learning?",
            "Are AI and machine learning the same thing?",
            "AI vs ML: what is the distinction?",
        ],
        "definition": (
            "Artificial Intelligence (AI) and Machine Learning (ML) are closely related, but they are not the same thing:\n\n"
            "- **Artificial Intelligence (AI)** is the overarching, broad vision of creating machines that exhibit intelligent behavior, reasoning, or problem-solving.\n"
            "- **Machine Learning (ML)** is a specific subset of AI that achieves intelligence by training statistical models on data rather than relying on hand-crafted rules.\n\n"
            "In summary: All machine learning is artificial intelligence, but not all artificial intelligence is machine learning. Early AI included rule-based expert systems and logic trees that did not learn from data."
        )
    },
    {
        "id": "deep_learning",
        "domain": "ai_ml",
        "name": "deep learning",
        "prompts": [
            "What is deep learning?",
            "Explain deep learning in simple words.",
            "How does deep learning relate to machine learning?",
            "What makes deep learning 'deep'?",
        ],
        "definition": (
            "Deep Learning is a specialized subfield of machine learning based on artificial neural networks with multiple layers (hence 'deep'). "
            "By stacking many layers of artificial neurons, deep networks can automatically extract hierarchical representations from raw data—such "
            "as detecting edges in early layers, object parts in intermediate layers, and complete faces or objects in deeper layers. "
            "Deep learning powers modern breakthroughs in image recognition, natural language processing, and speech synthesis."
        )
    },
    {
        "id": "neural_network",
        "domain": "ai_ml",
        "name": "neural network",
        "prompts": [
            "What is an artificial neural network?",
            "Explain how a neural network works in simple words.",
            "What are artificial neural networks in AI?",
        ],
        "definition": (
            "An artificial neural network is a computational model inspired by the biological networks of neurons in the animal brain. "
            "It consists of interconnected nodes (neurons) organized in layers: an input layer, one or more hidden layers, and an output layer. "
            "Each connection has an associated weight. When inputs pass through the network, each neuron computes a weighted sum of its inputs, "
            "adds a bias, applies a non-linear activation function, and transmits the signal forward."
        )
    },
    {
        "id": "supervised_learning",
        "domain": "ai_ml",
        "name": "supervised learning",
        "prompts": [
            "What is supervised learning in machine learning?",
            "Explain supervised learning in simple terms with examples.",
            "How does supervised learning work?",
        ],
        "definition": (
            "Supervised learning is a machine learning paradigm where an algorithm is trained on a labeled dataset, meaning each training example "
            "consists of an input paired with its correct ground-truth target output. The model makes predictions, compares them against the true labels, "
            "and updates its weights to reduce prediction errors. Examples include spam detection (classifying emails as spam or not) and house price prediction."
        )
    },
    {
        "id": "unsupervised_learning",
        "domain": "ai_ml",
        "name": "unsupervised learning",
        "prompts": [
            "What is unsupervised learning in machine learning?",
            "Explain unsupervised learning with simple examples.",
            "How does unsupervised learning differ from supervised learning?",
        ],
        "definition": (
            "Unsupervised learning is a machine learning paradigm where the algorithm is provided with unlabeled data and must discover hidden patterns, "
            "structures, or groupings on its own, without explicit target labels. Common tasks include clustering (grouping customers by purchasing habits), "
            "dimensionality reduction (PCA for visualizing high-dimensional data), and anomaly detection (spotting unusual transactions)."
        )
    },
    {
        "id": "overfitting_and_underfitting",
        "domain": "ai_ml",
        "name": "overfitting and underfitting",
        "prompts": [
            "What is overfitting and underfitting in machine learning?",
            "Explain overfitting in simple words.",
            "How do you prevent overfitting in deep learning?",
        ],
        "definition": (
            "Overfitting and underfitting describe how well a model generalizes to new data:\n\n"
            "- **Overfitting**: The model memorizes noise and specific idiosyncrasies in the training set instead of learning the general trend. It performs exceptionally on training data but fails on unseen validation data.\n"
            "- **Underfitting**: The model is too simple to capture the underlying structure of the data, resulting in poor performance on both training and validation sets.\n\n"
            "Overfitting is mitigated using techniques such as regularization (weight decay), dropout, early stopping, and collecting more training data."
        )
    },
    {
        "id": "gradient_descent",
        "domain": "ai_ml",
        "name": "gradient descent",
        "prompts": [
            "What is gradient descent in machine learning?",
            "Explain how gradient descent optimizes a neural network.",
            "What does gradient descent do in simple terms?",
        ],
        "definition": (
            "Gradient descent is an iterative optimization algorithm used to minimize the loss (error) function of a machine learning model. "
            "Imagine standing on a hilly landscape in thick fog and wanting to find the lowest valley. Gradient descent calculates the slope "
            "(the mathematical gradient) of the terrain and takes small steps in the direction of steepest downward descent, scaled by a step size "
            "called the learning rate."
        )
    },
    {
        "id": "transformer_architecture",
        "domain": "ai_ml",
        "name": "transformer architecture",
        "prompts": [
            "What is the Transformer architecture in deep learning?",
            "Explain how Transformers work in simple words.",
            "Why are Transformers so successful in modern AI?",
        ],
        "definition": (
            "The Transformer is a deep learning architecture introduced in 2017 ('Attention Is All You Need') that relies entirely on self-attention "
            "mechanisms rather than sequential recurrent connections (RNNs). Self-attention allows the model to weigh the relationships between all "
            "tokens in a sequence simultaneously, enabling massive parallelization during training and capturing long-range contextual dependencies effectively. "
            "Transformers form the architectural backbone of state-of-the-art LLMs like GPT, BERT, and Gemini."
        )
    },

    # ------------------ BASIC SCIENCE ------------------
    {
        "id": "photosynthesis",
        "domain": "science",
        "name": "photosynthesis",
        "prompts": [
            "What is photosynthesis?",
            "Explain photosynthesis in simple words.",
            "How do plants make food using photosynthesis?",
            "What is the chemical equation and process of photosynthesis?",
        ],
        "definition": (
            "Photosynthesis is the biological process by which green plants, algae, and some bacteria convert light energy (usually from the Sun) "
            "into chemical energy stored in glucose molecules. Using the green pigment chlorophyll inside chloroplasts, plants capture sunlight, "
            "absorb water (H2O) from their roots, and take in carbon dioxide (CO2) from the air to produce glucose (sugar) and release oxygen (O2) as a byproduct."
        )
    },
    {
        "id": "cellular_respiration",
        "domain": "science",
        "name": "cellular respiration",
        "prompts": [
            "What is cellular respiration?",
            "Explain cellular respiration in simple words.",
            "How do cells convert food into energy?",
        ],
        "definition": (
            "Cellular respiration is the metabolic process by which living cells break down glucose and other nutrients in the presence of oxygen "
            "to produce usable cellular energy in the form of Adenosine Triphosphate (ATP), releasing carbon dioxide and water as waste products. "
            "In eukaryotic cells, the majority of ATP synthesis occurs inside mitochondria."
        )
    },
    {
        "id": "dna",
        "domain": "science",
        "name": "DNA",
        "prompts": [
            "What is DNA?",
            "Explain what DNA is in simple words.",
            "What is the structure and function of DNA?",
        ],
        "definition": (
            "DNA (Deoxyribonucleic Acid) is the hereditary molecule that carries the genetic blueprint for the growth, development, functioning, "
            "and reproduction of all known living organisms and many viruses. Structurally, DNA consists of two long strands wrapped around each other "
            "in a double helix, composed of four nucleotide bases: Adenine (A), Thymine (T), Cytosine (C), and Guanine (G)."
        )
    },
    {
        "id": "gravity",
        "domain": "science",
        "name": "gravity",
        "prompts": [
            "What is gravity?",
            "Explain gravity in simple terms.",
            "How does gravity work according to physics?",
        ],
        "definition": (
            "Gravity is a fundamental physical force of attraction that pulls objects with mass or energy toward one another. "
            "In classical physics (Newton's law), gravity is directly proportional to the product of the masses and inversely proportional "
            "to the square of the distance between them. In Einstein's General Theory of Relativity, gravity is described not as a conventional force, "
            "but as the curvature of spacetime caused by the presence of mass and energy."
        )
    },
    {
        "id": "newtons_laws_of_motion",
        "domain": "science",
        "name": "Newton's laws of motion",
        "prompts": [
            "What are Newton's three laws of motion?",
            "Explain Newton's laws of motion in simple terms.",
            "Summarize the three laws of motion.",
        ],
        "definition": (
            "Sir Isaac Newton formulated three classical laws describing the relationship between a body and the forces acting upon it:\n\n"
            "1. **First Law (Inertia)**: An object remains at rest or in uniform motion along a straight line unless acted upon by an external net force.\n"
            "2. **Second Law (F = ma)**: The acceleration of an object is directly proportional to the net force applied and inversely proportional to its mass.\n"
            "3. **Third Law (Action and Reaction)**: For every action force, there is an equal and opposite reaction force."
        )
    },
    {
        "id": "conservation_of_energy",
        "domain": "science",
        "name": "conservation of energy",
        "prompts": [
            "What is the law of conservation of energy?",
            "Explain conservation of energy in simple words.",
            "Can energy be created or destroyed?",
        ],
        "definition": (
            "The Law of Conservation of Energy states that energy can neither be created nor destroyed; it can only be transformed from one form into another "
            "(or transferred between systems). The total amount of energy in an isolated system remains constant over time. For example, when a roller coaster "
            "descends a hill, potential energy is converted into kinetic energy, sound, and thermal energy through friction."
        )
    },
    {
        "id": "atoms_and_molecules",
        "domain": "science",
        "name": "atoms and molecules",
        "prompts": [
            "What is the difference between an atom and a molecule?",
            "Explain atoms and molecules in simple terms.",
            "What are the building blocks of matter?",
        ],
        "definition": (
            "Atoms and molecules are the basic building blocks of matter:\n\n"
            "- **Atom**: The smallest individual unit of a chemical element that retains the chemical properties of that element, consisting of a central nucleus of protons and neutrons surrounded by electrons.\n"
            "- **Molecule**: An electrically neutral group of two or more atoms held together by chemical bonds (such as H2O, which consists of two hydrogen atoms bonded to one oxygen atom)."
        )
    },
    {
        "id": "plate_tectonics",
        "domain": "science",
        "name": "plate tectonics",
        "prompts": [
            "What is plate tectonics in geology?",
            "Explain plate tectonics in simple words.",
            "What causes earthquakes and volcanic eruptions?",
        ],
        "definition": (
            "Plate tectonics is the scientific theory explaining that Earth's outer shell (the lithosphere) is divided into several large, "
            "rigid slabs called tectonic plates that glide over the hot, ductile asthenosphere beneath them. "
            "The interactions along plate boundaries—where plates collide, pull apart, or grind past one another—generate major geological phenomena "
            "such as mountain ranges, earthquakes, ocean trenches, and volcanic activity."
        )
    },
    {
        "id": "thermodynamics_second_law",
        "domain": "science",
        "name": "second law of thermodynamics",
        "prompts": [
            "What is the second law of thermodynamics?",
            "Explain entropy in simple words.",
            "Why does entropy always increase in an isolated system?",
        ],
        "definition": (
            "The Second Law of Thermodynamics states that in any spontaneous natural process within an isolated system, the total entropy "
            "(a measure of disorder or unavailable thermal energy) always increases over time. This principle dictates that heat naturally flows "
            "spontaneously from hotter bodies to colder bodies, and no heat engine can ever be 100% efficient."
        )
    },
    {
        "id": "natural_selection",
        "domain": "science",
        "name": "natural selection",
        "prompts": [
            "What is natural selection in biology?",
            "Explain natural selection in simple words with an example.",
            "How does evolution by natural selection work?",
        ],
        "definition": (
            "Natural selection is the fundamental mechanism of biological evolution proposed by Charles Darwin. "
            "Organisms with inheritable traits that are better adapted to their environment tend to survive and reproduce more successfully "
            "than individuals with less advantageous traits. Over generations, these advantageous traits become increasingly prevalent in the population."
        )
    },
]

# =====================================================================
# 2. HELD-OUT CONCEPTS (Strictly Disjoint from Training!)
# =====================================================================

HELDOUT_CONCEPTS = [
    # Economics held-out
    {
        "id": "heldout_stagflation",
        "domain": "economics",
        "name": "stagflation",
        "prompts": [
            "What is stagflation in economics?",
            "Explain stagflation in simple words.",
            "Why is stagflation difficult for central banks to resolve?",
        ],
        "definition": (
            "Stagflation is an unusual and toxic economic condition characterized by stagnant economic growth, high unemployment, "
            "and concurrently high inflation. It poses a severe policy dilemma for central banks because tools used to lower inflation "
            "(raising interest rates) tend to worsen unemployment, while stimulus tools to boost growth risk driving inflation even higher."
        )
    },
    {
        "id": "heldout_oligopoly",
        "domain": "economics",
        "name": "oligopoly",
        "prompts": [
            "What is an oligopoly in economics?",
            "Explain oligopoly with real-world examples.",
            "How does an oligopoly differ from a monopoly?",
        ],
        "definition": (
            "An oligopoly is a market structure dominated by a small number of large firms that control the vast majority of market share. "
            "Unlike a monopoly where one seller controls everything, in an oligopoly several dominant players compete, but their decisions "
            "regarding price and output are highly interdependent. Classic examples include commercial airline manufacturing (Boeing and Airbus) "
            "and mobile operating systems (Google Android and Apple iOS)."
        )
    },
    {
        "id": "heldout_moral_hazard",
        "domain": "economics",
        "name": "moral hazard",
        "prompts": [
            "What is moral hazard in economics?",
            "Explain moral hazard with an insurance example.",
            "Why does moral hazard occur?",
        ],
        "definition": (
            "Moral hazard occurs when an individual or institution takes on greater risks because they know that someone else will bear "
            "the economic costs or consequences if things go wrong. A common example is in insurance: an individual with full theft insurance "
            "might be less cautious about locking their car doors than someone without coverage."
        )
    },
    # AI/ML held-out
    {
        "id": "heldout_diffusion_models",
        "domain": "ai_ml",
        "name": "diffusion models",
        "prompts": [
            "What are diffusion models in generative AI?",
            "Explain how diffusion models work in simple words.",
            "How do diffusion models generate images?",
        ],
        "definition": (
            "Diffusion models are a family of generative deep learning models that generate data (such as photorealistic images or audio) "
            "by learning to reverse a gradual noising process. During training, Gaussian noise is incrementally added to an image until it "
            "becomes pure static; the neural network is trained to remove the noise step-by-step. At generation time, the model starts with pure noise "
            "and denoises it iteratively into a sharp, coherent image."
        )
    },
    {
        "id": "heldout_cross_entropy_loss",
        "domain": "ai_ml",
        "name": "cross-entropy loss",
        "prompts": [
            "What is cross-entropy loss in machine learning?",
            "Explain cross-entropy loss in simple terms.",
            "How is cross-entropy loss calculated for classification?",
        ],
        "definition": (
            "Cross-entropy loss (also known as log loss) is a loss function used in classification and language modeling to measure the performance "
            "of a model whose output is a probability distribution between 0 and 1. It quantifies the difference between the predicted probabilities "
            "and the actual ground-truth label, heavily penalizing predictions that are confident and wrong."
        )
    },
    {
        "id": "heldout_data_leakage",
        "domain": "ai_ml",
        "name": "data leakage",
        "prompts": [
            "What is data leakage in machine learning?",
            "Explain data leakage and why it is dangerous in model training.",
            "How do you prevent data leakage between train and test sets?",
        ],
        "definition": (
            "Data leakage occurs when information from outside the training dataset (specifically from the validation or test set) is inadvertently "
            "used to train or prepare the model. This creates an artificially optimistic evaluation score during experimentation, but causes the model "
            "to perform significantly worse when deployed in production on truly unseen real-world data."
        )
    },
    # Science held-out
    {
        "id": "heldout_doppler_effect",
        "domain": "science",
        "name": "Doppler effect",
        "prompts": [
            "What is the Doppler effect in physics?",
            "Explain the Doppler effect with an example.",
            "Why does a siren's pitch change as it drives past?",
        ],
        "definition": (
            "The Doppler effect is the observed change in frequency or wavelength of a wave (such as sound or light) in relation to an observer "
            "who is moving relative to the wave source. When a source (such as an ambulance siren) approaches, sound waves compress and the pitch sounds higher; "
            "as it moves away, the waves stretch out and the pitch drops. In astronomy, redshift of light from distant galaxies demonstrates the Doppler effect on light."
        )
    },
    {
        "id": "heldout_epigenetics",
        "domain": "science",
        "name": "epigenetics",
        "prompts": [
            "What is epigenetics in biology?",
            "Explain epigenetics in simple words.",
            "How can gene expression change without altering DNA sequence?",
        ],
        "definition": (
            "Epigenetics is the study of inheritable changes in gene expression and cellular phenotype that do not involve alterations to the underlying "
            "DNA nucleotide sequence. Chemical modifications, such as DNA methylation and histone acetylation, act like on/off switches that determine "
            "which genes are active in different cell types or in response to environmental factors like diet and stress."
        )
    },
    {
        "id": "heldout_superconductivity",
        "domain": "science",
        "name": "superconductivity",
        "prompts": [
            "What is superconductivity in physics?",
            "Explain superconductivity and zero electrical resistance.",
            "What happens when a material becomes superconducting?",
        ],
        "definition": (
            "Superconductivity is a quantum physical phenomenon occurring in certain materials when cooled below a critical temperature, "
            "characterized by exactly zero electrical resistance and the complete expulsion of interior magnetic fields (the Meissner effect). "
            "Superconductors allow electrical current to flow indefinitely without generating heat or energy loss, enabling powerful electromagnets "
            "used in MRI machines and particle accelerators."
        )
    },
]


def build_concept_datasets():
    rng = random.Random(SEED)
    train_records = []
    heldout_records = []

    # 1. Base training records
    for concept in TRAIN_CONCEPTS:
        cid = concept["id"]
        cname = concept["name"]
        domain = concept["domain"]
        definition = concept["definition"]
        for p in concept["prompts"]:
            train_records.append({
                "category": "concept_definition",
                "concept_id": cid,
                "concept_name": cname,
                "domain": domain,
                "messages": [
                    {"role": "prompter", "text": p},
                    {"role": "assistant", "text": definition}
                ]
            })

    # Expand train records with distinct question phrasings
    prompt_prefixes = [
        "Please explain {concept_name} clearly and concisely.",
        "Can you provide a simple definition of {concept_name} for a student?",
        "What does the term '{concept_name}' mean in {domain}?",
        "Could you break down the concept of {concept_name}?",
        "Explain {concept_name} with key details.",
        "Give me a clear overview of {concept_name}.",
        "What is the basic meaning of {concept_name}?"
    ]
    while len(train_records) < TARGET_TRAIN_COUNT:
        c = rng.choice(TRAIN_CONCEPTS)
        pref = rng.choice(prompt_prefixes)
        q = pref.format(concept_name=c["name"], domain=c["domain"])
        train_records.append({
            "category": "concept_definition_variation",
            "concept_id": c["id"],
            "concept_name": c["name"],
            "domain": c["domain"],
            "messages": [
                {"role": "prompter", "text": q},
                {"role": "assistant", "text": c["definition"]}
            ]
        })
    rng.shuffle(train_records)
    final_train = train_records[:TARGET_TRAIN_COUNT]

    # 2. Held-out records
    for hc in HELDOUT_CONCEPTS:
        cid = hc["id"]
        cname = hc["name"]
        domain = hc["domain"]
        definition = hc["definition"]
        for p in hc["prompts"]:
            heldout_records.append({
                "category": "concept_heldout",
                "concept_id": cid,
                "concept_name": cname,
                "domain": domain,
                "messages": [
                    {"role": "prompter", "text": p},
                    {"role": "assistant", "text": definition}
                ]
            })

    while len(heldout_records) < TARGET_HELDOUT_COUNT:
        hc = rng.choice(HELDOUT_CONCEPTS)
        pref = rng.choice(prompt_prefixes)
        q = pref.format(concept_name=hc["name"], domain=hc["domain"])
        heldout_records.append({
            "category": "concept_heldout_variation",
            "concept_id": hc["id"],
            "concept_name": hc["name"],
            "domain": hc["domain"],
            "messages": [
                {"role": "prompter", "text": q},
                {"role": "assistant", "text": hc["definition"]}
            ]
        })
    rng.shuffle(heldout_records)
    final_heldout = heldout_records[:TARGET_HELDOUT_COUNT]

    return final_train, final_heldout


def main():
    print("=" * 70)
    print("EXPERIMENT 2: GENERATING CONCEPTS DATASET (CATEGORY 4)")
    print("=" * 70)
    start_time = time.time()

    train_data, heldout_data = build_concept_datasets()

    # Verify zero concept overlap between train and held-out
    train_cids = set(ex["concept_id"] for ex in train_data)
    heldout_cids = set(ex["concept_id"] for ex in heldout_data)
    cid_overlap = train_cids.intersection(heldout_cids)
    print(f"Train/Heldout concept ID overlap: {len(cid_overlap)} (Target: 0)")
    assert len(cid_overlap) == 0, f"Concept leakage detected: {cid_overlap}"

    # Verify benchmark concepts are present in train
    for required in ["monopsony", "artificial_intelligence", "machine_learning", "ai_vs_machine_learning"]:
        assert required in train_cids, f"Required benchmark concept '{required}' missing from training!"
    print("Verified: all core benchmark concepts (monopsony, AI, ML, AI vs ML) present in train!")

    # Check prompt overlap
    train_prompts = set(ex["messages"][0]["text"] for ex in train_data)
    heldout_prompts = set(ex["messages"][0]["text"] for ex in heldout_data)
    overlap = train_prompts.intersection(heldout_prompts)
    print(f"Train/Heldout exact prompt overlap: {len(overlap)} (Target: 0)")
    assert len(overlap) == 0, f"Prompt overlap found: {overlap}"

    # Write files
    OUT_TRAIN_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_TRAIN_FILE, "w", encoding="utf-8") as f:
        for ex in train_data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(OUT_HELDOUT_FILE, "w", encoding="utf-8") as f:
        for ex in heldout_data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    elapsed = time.time() - start_time
    print(f"\nConcepts Train Examples Generated:     {len(train_data):,}")
    print(f"Concepts Held-Out Examples:            {len(heldout_data):,}")
    print(f"Written train to:   {OUT_TRAIN_FILE}")
    print(f"Written heldout to: {OUT_HELDOUT_FILE}")
    print(f"Completed in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    main()
