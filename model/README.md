<h2 align="center"> BiCANet </h2>

Exploring contextual information in convolutional neural networks (CNNs) has gained substantial attention in recent years for semantic segmentation. 

Bi-directional Contextual Aggregating Network, called BiCANet, is an architecture for semantic segmentation. 

Unlike previous approaches that encode context in feature space, BiCANet aggregates contextual cues from a categorical perspective, which mainly consists of three parts: contextual condensed projection block (CCPB), bidirectional context interaction block (BCIB), and muti-scale contextual fusion block (MCFB).

More specifically, CCPB learns a category-based mapping through a split-transform-merge architecture, which condenses contextual cues with different receptive fields from intermediate layer. BCIB, on the other hand, employs dense skipped-connections to enhance the classlevel context exchanging. Finally, MCFB integrates multi-scale contextual cues by investigating short- and long-ranged spatial dependencies.

<p align="center">
  <img width="623" height="270" src="https://github.com/bcd8697/BiCANet/blob/main/images/architecture.png">
</p>
