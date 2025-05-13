# Potential Citations Needed

This document lists sentences and concepts from the LaTeX files that may require citations. Please review each item to determine if a citation is necessary based on your field's standards and the specific context.

## Thesis/introduction.tex

-   **Sentence:** "Efficient transportation is a critical component of university campus life, impacting student accessibility, operational costs, and environmental sustainability."
    **Reason:** General statement about the importance of transportation in university contexts; might benefit from a supporting citation.
-   **Sentence:** "These include the inherent complexity of optimizing routes for a dispersed population with varying demand, the need for computationally tractable solutions for large-scale networks, and the impact of real-world data imperfections such as outliers."
    **Reason:** Stating these as known challenges might need backing from literature.
-   **Sentence:** "...appropriate sparsity can reveal underlying network structures crucial for optimization."
    **Reason:** Claim about sparsity benefits could warrant a citation from graph theory or network analysis literature.
-   **Sentence:** "Furthermore, to effectively group students into viable bus routes, we evaluate the performance of several state-of-the-art graph clustering algorithms – namely Spectral Clustering, the Leiden Algorithm, and Multi-view Anchor Graph-based Clustering (MVAGC)."
    **Reason:** Calling algorithms "state-of-the-art" might need justification or individual citations for each algorithm's foundational paper.
-   **Sentence:** "The problem of vehicle routing and transportation network optimization has been extensively studied, with a rich body of literature focusing on various methodologies."
    **Reason:** General claim about extensive study; could cite a review paper on vehicle routing or transportation optimization.
-   **Sentence:** "Graph theory provides a natural and powerful framework for modeling transportation networks, where locations are represented as vertices and travel segments as edges."
    **Reason:** States a common application of graph theory; might benefit from a foundational citation on graph theory in network modeling or a relevant textbook.
-   **Sentence:** "While complete graphs offer comprehensive connectivity, their $O(N^2)$ complexity is prohibitive for large datasets."
    **Reason:** States a known property and limitation of complete graphs.
-   **Sentence:** "Consequently, sparse graph construction methods such as K-Nearest Neighbors (KNN) graphs, Delaunay Triangulation, and Gabriel Graphs are widely adopted."
    **Reason:** Claims these methods are "widely adopted." Foundational papers or review articles discussing their adoption in transportation/network contexts would be appropriate.
-   **Sentence:** "Graph clustering, or community detection, seeks to identify groups of vertices that are densely connected internally while being sparsely connected to the rest of the graph."
    **Reason:** Definition of graph clustering/community detection; could cite a foundational paper or review on the topic.
-   **Sentence:** "Furthermore, outlier detection methods are increasingly recognized as important pre-processing steps in network analysis."
    **Reason:** General statement about the growing importance of outlier detection in network analysis; a supporting citation would strengthen this.

## Thesis/abstract.tex

-   **Sentence:** "...sparse graph representations are crucial for computational tractability and improved routing solutions compared to a complete graph."
    **Reason:** While a finding of the thesis, the general principle that sparse graphs can offer advantages is citable.
-   **Concept:** Mention of specific algorithms (Spectral Clustering, Leiden, MVAGC, KNN outlier detection).
    **Reason:** Implies these are established methods; ensure citations are present in the main body of the thesis where these are detailed.

## Thesis/conclusions.tex

-   **Sentence:** "Our research demonstrates that graph-based approaches offer powerful tools for transportation network optimization."
    **Reason:** General statement that could be supported by broader literature on graph theory in operations research or transportation.
-   **Sentence:** "The comparative analysis of different graph construction methods revealed that sparse representations such as Delaunay triangulation, Gabriel graphs, and K-Nearest Neighbors significantly outperform complete graphs both in computational efficiency and solution quality."
    **Reason:** While your finding, the general concept that sparse graphs can be better for efficiency/quality in certain contexts might have prior art.
-   **Sentence:** "Specifically, the Gabriel graph emerged as the most effective representation, balancing connectivity and sparsity to capture essential spatial relationships while eliminating redundant edges."
    **Reason:** The properties of Gabriel graphs are established; citing their original description or relevant properties might be appropriate.
-   **Sentence:** "...Leiden algorithm consistently produced the highest quality solutions... This performance validates the algorithm's refinement mechanism that ensures well-connected communities while optimizing modularity."
    **Reason:** The description of the Leiden algorithm's mechanism should be attributed to its source.
-   **Sentence:** "The integration of K-Nearest Neighbor distance-based outlier detection as a preprocessing step further enhanced performance..."
    **Reason:** The KNN outlier detection method itself is an existing technique and should be cited.
-   **Sentence:** "First, we have established a comprehensive framework for comparing different graph construction methods in the context of transportation planning, demonstrating the advantages of sparse representations for large-scale networks."
    **Reason:** If similar frameworks or findings exist, they should be acknowledged.
-   **Sentence:** "Second, our systematic evaluation of clustering algorithms highlights the effectiveness of community detection approaches for identifying optimal bus routes, with particular emphasis on the Leiden algorithm's capabilities for transportation applications."
    **Reason:** If community detection or the Leiden algorithm has been previously applied to transportation problems, that work is citable.
-   **Sentence:** "Third, the implementation of outlier detection as a preprocessing step represents a novel enhancement to the transportation planning pipeline, offering significant benefits with minimal computational overhead."
    **Reason:** Check if outlier detection has been used in transportation planning before, even if not in this exact pipeline, and cite if so.
-   **Section:** "Future Work" (mentions Dynamic routing, Multi-objective optimization, Machine learning integration, etc.)
    **Reason:** Each of these is an established research area. Consider citing a key paper or review for each to set the context for your proposed future work.

## Thesis/appendix.tex

-   **Concept:** Leiden Algorithm Pseudocode.
    **Reason:** While describing the algorithm, ensure the original Leiden algorithm paper is cited (it appears to be in `main.tex`'s bibliography). Verify the pseudocode accurately reflects the cited source or clearly note any modifications.

## Thesis/experiments.tex

-   **Sentence context:** "...computed using Dijkstra's algorithm (Section~\ref{subsec:DijkstrasAlgorithm})."
    **Reason:** Dijkstra's algorithm is standard and should be cited where it's first introduced/described in detail (likely `theory.tex` or `method.tex`).
-   **Sentence:** "As expected, the sparse graphs significantly reduce the number of edges compared to the Complete Graph, making subsequent processing more computationally tractable."
    **Reason:** This is a general property discussed in graph theory when comparing dense and sparse graphs.
-   **Sentence context:** "Clustering the Complete Graph (using the Leiden algorithm as described in Section~\ref{subsec:clustering_complete}) serves as a theoretical baseline."
    **Reason:** The Leiden algorithm should be cited.
-   **Sentence:** "The high computational cost and tendency towards imbalanced clusters motivate the evaluation of clustering algorithms on the sparse graph representations."
    **Reason:** The idea that dense graphs can lead to imbalanced clusters or computational issues might be a known challenge in the literature.
-   **Sentence:** "The outlier detection process was designed to identify and remove anomalous student locations that could potentially skew clustering results and lead to suboptimal routes."
    **Reason:** This describes a general benefit/purpose of outlier detection.
-   **Sentence:** "The results demonstrate that outlier detection consistently improves transportation network efficiency across all graph construction methods and clustering algorithms."
    **Reason:** If this is a known general benefit of outlier detection in such problems, it could be supported by literature.
-   **Sentence:** "These findings highlight the importance of data preprocessing in transportation network optimization. The KNN distance-based outlier detection method provided a computationally efficient way to improve the quality of the resulting clusters... For real-world transportation planning scenarios, this suggests that investing in outlier detection and data cleaning can yield substantial returns..."
    **Reason:** Strong claim about general applicability and benefits. Broader claims about "real-world transportation planning" often benefit from citations showing similar findings or recommendations.
-   **Section:** Discussion (e.g., "Sparse graph representations such as Delaunay, Gabriel, and KNN are essential for computational tractability...")
    **Reason:** General claims made here might be supported by existing literature in graph theory or network analysis, beyond just your own results.

## Thesis/method.tex

-   **Sentence:** "These points were created by applying a Gaussian distribution based on the actual population data for each district in Izmir."
    **Reason:** If the method of using Gaussian distribution for synthetic population generation is based on existing methodologies, or if the population data source itself needs to be cited.
-   **Concept:** Complete Graph as a baseline model.
    **Reason:** Its application as a baseline in network analysis might have citable precedents.
-   **Concept:** Delaunay triangulation.
    **Reason:** A well-defined mathematical construct; its original source or a good explanatory text should be cited when introduced.
-   **Concept:** Gabriel graph.
    **Reason:** Has a formal definition and origin that should be cited.
-   **Concept:** K-Nearest Neighbors (KNN) graph.
    **Reason:** KNN is a standard technique; cite its origin.
-   **Concept:** Leiden algorithm and modularity optimization.
    **Reason:** Should be cited.
-   **Concept:** Spectral clustering, use of normalized Laplacian matrix, eigendecomposition.
    **Reason:** Established techniques; cite foundational work.
-   **Concept:** Leiden algorithm implementation (if modified from original).
    **Reason:** Cite original Leiden. If custom quality function is inspired by or similar to other approaches, cite those.
-   **Concept:** MVAGC implementation (especially if adapted).
    **Reason:** Cite original MVAGC algorithm.
-   **Concept:** Dijkstra's algorithm.
    **Reason:** Needs citation.
-   **Concept:** K-Nearest Neighbor (KNN) distance method for outlier detection.
    **Reason:** Known method; should be cited where detailed (you mention "detailed theoretically in Section~\ref{subsec:KNNDistanceOutlier}", so the citation should be there).

## Thesis/theory.tex

**General Note:** This chapter introduces established concepts. Every major concept/algorithm described here requires a citation to its origin or a standard textbook/reference work.

-   **Concept:** Basic Graph Theory definitions (Section \ref{se:BasicDefinitions}).
    **Action:** Cite a general graph theory textbook.
-   **Concept:** Complete Graph definition and properties (Section \ref{se:GraphConstructionMethodsAndSparsity}).
    **Action:** Cite relevant graph theory source.
-   **Concept:** K-Nearest Neighbors (KNN) Graph (Section \ref{se:GraphConstructionMethodsAndSparsity}).
    **Action:** Cite foundational work on KNN.
-   **Concept:** Delaunay Triangulation (Section \ref{se:GraphConstructionMethodsAndSparsity}).
    **Action:** Cite original work or defining text (e.g., Delaunay's paper, or a computational geometry textbook).
-   **Concept:** Gabriel Graph (Section \ref{se:GraphConstructionMethodsAndSparsity}).
    **Action:** Cite original work (Gabriel and Sokal) or defining text.
-   **Concept:** Graph-based Clustering (Section \ref{se:GraphBasedClusterings}).
    **Action:** Cite overview papers on graph clustering or a relevant textbook.
-   **Concept:** Spectral Clustering, Laplacian matrix, k-means application (Section \ref{subsec:SpectralClustering}).
    **Action:** Cite foundational papers (e.g., Ng, Jordan, and Weiss; Shi and Malik) or textbook explanation.
-   **Concept:** Leiden Algorithm and Modularity (Section \ref{subsec:LeidenAlgorithm}).
    **Action:** Original Leiden paper is cited (`\cite{leiden}`). Consider citing Louvain for context (as Leiden improves upon it). The modularity definition also has origins (e.g., Newman and Girvan).
-   **Concept:** Multi-view Anchor Graph-based Clustering (MVAGC) (Section \ref{subsec:MVAGC}).
    **Action:** Cite the original paper(s) introducing MVAGC.
-   **Concept:** Shortest Path Algorithms (Section \ref{se:ShortestPathAlgorithms}).
    **Action:** Cite a general algorithms textbook.
-   **Concept:** Dijkstra's Algorithm (Section \ref{subsec:DijkstrasAlgorithm}).
    **Action:** Cite Dijkstra's original paper or a standard algorithms textbook.
-   **Concept:** Outlier Detection (Section \ref{se:OutlierDetection}).
    **Action:** Cite overview papers on outlier detection.
-   **Concept:** K-Nearest Neighbor (KNN) Distance Outlier Detection (Section \ref{subsec:KNNDistanceOutlier}).
    **Action:** Foundational work on KNN outlier detection is cited (`\cite{knn_outlier}`). 