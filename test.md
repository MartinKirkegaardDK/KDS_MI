- preprocessing
	- filter small sentences
	- test/train/eval?
- steering vector (with other method)
- driver functions to run utilities (experiments)
- run experiments and save results
- from meeting:
	- other method for where to insert probes
		- maybe perplexity (loss on parallel data under steering)
	- evaluation of text after steering
		- language id model to see which language text is (this would also check if output is coherent?) (in martin words: run/train language model
		to classify which language text is (is it english or danish) on generated text form the steered model output. e.i we put english in, steer it to danish, run classification model on output to quantify if this text is danish. )
		- (using parallel data) give it the first half of a lan1 text and measure loss on the continued text. then compare that to if the first half was lan2 but it has been steered towards lan1.
- idea from meeting:
	- can we recover lost performance on continual learning models with steering vectors



- layout of general presentation
	- activations for different data
		- PCA
		- Probes
	- computation of steering vectors
		- euclidian distance between them
		- PCA of steering vectors
		- loss of them on different layers
	- demonstration of steering
	- classification score of steering vectors



- for meeting
	- plots for
		- language classification for different lambda
		- loss for different lambda
			- weird stuff on layer 6 and layer 7
				- memory management?
	- PCA for different layer
	- Models for more languages
	- 