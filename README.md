# ProjectSchemasAnnotator
Project developed as part of the SCHEMAS project (financed by the Serbian ministry of Science). 
Two-Stage Machine Learning Pipeline for Schema Annotation: Boundary Detection and Multi-Label Classification
Introduction

Cognitive linguistics, over the past few decades, has foregrounded the intricate interplay between conceptual structures and linguistic expressions, demonstrating that language is not an autonomous formal system but rather one deeply rooted in embodied cognition (Evans & Green, 2006; Langacker, 2008). From the early works of Lakoff, Johnson, and their contemporaries, it has become a foundational principle that meaning is grounded in sensory-motor experiences and shaped by patterns of bodily interaction with the environment (Johnson, 1987; Lakoff, 1987). Within this theoretical context, image schemas have emerged as a particularly salient concept: they are recurrent patterns of perception and movement that inform and structure more complex conceptualizations. Image schemas are understood as basic, preverbal gestalts that arise from embodied experience and recur across multiple conceptual domains, providing a scaffold upon which we build more elaborate metaphorical and abstract reasoning. While their role in conceptual meaning and metaphorical extension is well recognized, less attention has been paid to how image schemas are syntactically realized and how their underlying conceptual patterns might shape the structural patterns we observe in language. This gap not only poses theoretical questions but also presents a practical challenge for computational approaches to language—how might we design models that automatically detect these image-schematic patterns in naturally occurring texts? Addressing such a question opens up a fruitful area of inquiry: if image schemas are truly foundational to our conceptual apparatus, then examining their formal and syntactic realizations can yield deeper insights into how language is constructed, processed, and understood, including insights gleaned through machine learning (ML) methods.
At the core of much cognitive linguistic research lies the concept of the image schema: a dynamic, embodied pattern of experience that emerges through recurring sensorimotor interactions with our environment (Johnson, 1987; Lakoff, 1987; Mandler, 2004). Image schemas are not static mental pictures; rather, they are continuous and interactive patterns of experience, such as moving along a path, exerting force, or being contained within a bounded space. According to Johnson (1987), these schemas arise early in cognitive development as infants learn to balance their bodies, manipulate objects, perceive objects entering or leaving containers, and navigate through space. Such patterns become internalized as mental templates that can be metaphorically extended to structure more abstract conceptual domains. For example, the SOURCE-PATH-GOAL schema—one of the most frequently cited examples—originates in our repeated experience of physically moving from one location to another and then shapes our understanding of events, life journeys, and abstract conceptual progressions (Lakoff, 1987; Mandler, 2004).
Because image schemas are grounded in bodily experience, they are generally seen as universal or near-universal cognitive structures. They emerge from bodily interactions common to all humans, such as UP-DOWN arising from our vertical orientation or CONTAINER from our awareness of boundaries and enclosures (Johnson, 1987; Evans & Green, 2006). However, the particular linguistic and cultural elaborations of these schemas may vary, allowing some universality in conceptual grounding to coexist with linguistic diversity. Research in cognitive linguistics has focused heavily on how image schemas shape metaphorical mappings, conceptual blends, and lexical semantics, demonstrating their pervasive influence across numerous languages and conceptual domains.
Despite this rich body of work, the attention to image schemas has primarily concentrated on their conceptual and semantic dimensions. Cognitive linguists have extensively explored how image schemas inform meaning, conceptual metaphors, and reasoning (Lakoff, 1987; Johnson, 1987), but the question of how these conceptual patterns manifest syntactically—how they are encoded, hinted at, or reinforced by the grammatical structures of language—has received comparatively less systematic scrutiny. Although some early studies have hinted at possible correlations between syntactic constructions and underlying image-schematic structures (Clausner & Croft, 1999), the field as a whole has yet to fully engage with the challenge of mapping image schemas onto syntactic patterns. Moreover, a machine learning perspective on this problem—developing algorithms to detect and label potential image-schematic structures in text—remains largely unexplored.
The question at the heart of this study is how image schemas, as conceptual building blocks, are manifested in the syntactic fabric of language. If image schemas truly underpin a wide range of conceptual and linguistic phenomena, it follows that their influence might be detectable not only at the semantic or conceptual level but also in the structural arrangement of clauses, phrases, and sentences. For instance, we might ask whether certain image schemas correlate strongly with particular construction types, prepositional phrases, verb classes, or argument structures. Another avenue of inquiry might involve examining how force dynamics—an essential component of many image schemas—map onto grammatical patterns that encode agency, causation, and affectedness. Crucially, such mapping raises the possibility of training computational models to automate the detection of these schematic cues, offering a scalable approach to analyzing large corpora.
To approach this problem, one significant hurdle must be addressed: the relative lack of empirical research employing large-scale, corpus-driven methods to investigate the syntactic realizations of image schemas. Much of the seminal work in cognitive linguistics and image schema research has relied on introspective data, small sets of examples, or carefully constructed experimental stimuli (Johnson, 1987; Lakoff & Johnson, 1999). While these methods have yielded valuable theoretical insights, they do not easily lend themselves to demonstrating systematic correlations between underlying image schemas and particular syntactic constructions in naturally occurring language. There is a need for comprehensive, empirical corpus analyses that can test hypotheses about the systematic patterns through which image schemas might emerge in syntax and structure our linguistic expressions. Furthermore, integrating corpus-based annotations with a machine learning system offers a powerful synergy: the annotated corpus can train the model to recognize subtle syntactic cues associated with specific image schemas, thereby enabling deeper corpus analysis at scale.
Despite advances in corpus linguistics and the availability of extensive corpora, cognitive linguistic research has only recently begun to embrace large-scale empirical approaches more fully. While some recent studies have investigated metaphorical patterns, collocations, and semantic frames through corpus methods, image schema research remains somewhat behind this curve. Few studies have explicitly linked certain syntactic constructions to underlying image schemas based on corpus evidence, leaving this a relatively under-explored frontier in cognitive linguistics. As a result, we lack robust quantitative data that could either confirm or challenge existing theoretical claims about how deeply image schemas permeate language structure. This is precisely where computational tools—such as machine learning models for boundary detection, multi-label classification, or chunk-level annotation—can bridge the gap, offering systematic, large-scale evidence.
This lack of corpus-level research represents both a gap in the literature and a new opportunity. Establishing clear, reproducible correlations between image schemas and syntactic patterns would not only provide empirical support for theoretical claims but also facilitate cross-linguistic comparisons. Are certain image schemas universally associated with particular syntactic structures, or do languages vary widely in how they encode these conceptual patterns? Answering these questions would broaden our understanding of the language-cognition interface, shedding new light on the interplay between conceptual patterns and grammatical form. In tandem, machine learning approaches could be extended cross-linguistically, potentially detecting universal vs. language-specific features in how image schemas surface in syntax.
To address these concerns, this paper will draw on the SCHEMAS team project corpus, a specially curated dataset that has been assembled to facilitate the study of image schemas and their linguistic realizations. The SCHEMAS corpus brings together a range of carefully selected texts that have been annotated by trained researchers to reflect known or hypothesized image-schematic patterns. This corpus provides a unique resource for the present study, allowing for the systematic examination of syntactic patterns associated with particular schemas across a large and varied dataset. By leveraging this corpus, we seek to identify, quantify, and analyze the syntactic manifestations of image schemas in authentic language use. Moreover, an integral part of this research is the development of a machine learning model that will be trained on these annotated data, aiming to automate the detection of image-schematic cues and thereby scale the analysis to larger corpora or different text genres.
In the following sections, this paper will begin by surveying the theoretical landscape, situating image schemas within cognitive linguistics and reviewing relevant literature on their conceptual importance. It will then present a methodological framework for operationalizing image schemas in syntactic annotation and analysis. Drawing on the SCHEMAS corpus, the study will investigate the extent to which certain image schemas correlate with identifiable syntactic structures, describing the patterns that emerge and discussing their theoretical implications. Finally, it will consider the broader significance of these findings for our understanding of the relationship between embodied cognition and linguistic form, offering suggestions for future research that might build upon these insights.
By approaching image schemas from a corpus-based, syntactically oriented perspective—and integrating machine learning to detect and classify the schematic cues in text—this study aims to enrich the field’s understanding of how embodied conceptual patterns shape linguistic structure. It will illustrate that the influence of image schemas on language is not limited to semantics or metaphor; rather, these fundamental embodied concepts can also guide and constrain the way we construct our utterances. Such insights will underscore the importance of integrating theoretical claims with empirical corpus data while demonstrating the feasibility of a computational approach that leverages ML for large-scale image-schema detection. Ultimately, this convergence of corpus methods, linguistic theory, and machine learning will contribute to a more nuanced and comprehensive account of language as an embodied, cognitive, and structured phenomenon.

