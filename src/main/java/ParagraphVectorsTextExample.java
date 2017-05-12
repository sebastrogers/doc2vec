import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.AbstractCache;
import org.deeplearning4j.text.documentiterator.LabelsSource;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.ArrayList;

/**
 * This is example code for dl4j ParagraphVectors implementation. In this example we build distributed representation of all sentences present in training corpus.
 * However, you still use it for training on labelled documents, using sets of LabelledDocument and LabelAwareIterator implementation.
 *
 * *************************************************************************************************
 * PLEASE NOTE: THIS EXAMPLE REQUIRES DL4J/ND4J VERSIONS >= rc3.8 TO COMPILE SUCCESSFULLY
 * *************************************************************************************************
 *
 * @author raver119@gmail.com
 */
public class ParagraphVectorsTextExample {

	private static final Logger log = LoggerFactory.getLogger(ParagraphVectorsTextExample.class);

	public static void main(String[] args) throws Exception {
		ClassPathResource resource = new ClassPathResource("\\PreconceitoConvertido.txt");
		File file = resource.getFile();
		SentenceIterator iter = new BasicLineIterator(file);

		AbstractCache<VocabWord> cache = new AbstractCache<>();

		TokenizerFactory t = new DefaultTokenizerFactory();
		t.setTokenPreProcessor(new CommonPreprocessor());

		/*
             if you don't have LabelAwareIterator handy, you can use synchronized labels generator
              it will be used to label each document/sequence/line with it's own label.
              But if you have LabelAwareIterator ready, you can can provide it, for your in-house labels
		 */
		LabelsSource source = new LabelsSource("DOC_");

		ParagraphVectors vec = new ParagraphVectors.Builder()
				.minWordFrequency(1)
				.iterations(5)
				.epochs(1)
				.layerSize(100)
				.learningRate(0.025)
				.labelsSource(source)
				.windowSize(5)
				.iterate(iter)
				.trainWordVectors(false)
				.vocabCache(cache)
				.tokenizerFactory(t)
				.sampling(0)
				.build();

		vec.fit();

			
		double similarity2 = vec.similarity("DOC_1", "DOC_6780");
		log.info("3721/16393 ('sentença 1 .'/'decepcionado .') similarity: " + similarity2);

		//double similarity3 = vec.similarity("DOC_3", "DOC_4");
		//log.info("6348/3721 ('This is my case .'/'This is my way .') similarity: " + similarity3);

		// likelihood in this case should be significantly lower
		//double similarityX = vec.similarity("DOC_6", "DOC_7");
		//log.info("3721/9853 ('This is my way .'/'We now have one .') similarity: " + similarityX +
				//"(should be significantly lower)");
		
		/*
		for (String word : p) {
			//double similarity1 = vec.similarity("racismo", word);
			//log.info("9835/palavra ('O racismo é um câncer social .'/'" +word +".)" +"similarity: " + similarity1);
					
			double similarity2 = vec.similarity("DOC_3720", "DOC_16392");
			log.info("3721/16393 ('This is my way .'/'This is my work .') similarity: " + similarity2);

			double similarity3 = vec.similarity("DOC_6347", "DOC_3720");
			log.info("6348/3721 ('This is my case .'/'This is my way .') similarity: " + similarity3);

			// likelihood in this case should be significantly lower
			double similarityX = vec.similarity("DOC_3720", "DOC_9852");
			log.info("3721/9853 ('This is my way .'/'We now have one .') similarity: " + similarityX +
					"(should be significantly lower)");
		}
		
		*/
	}
}