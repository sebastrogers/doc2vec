import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;


public class Somatorio {

    public static void main(String[] args) {
        
        int indice = 0;
        
        ArrayList<String>  p = ler("ListaP1.txt");
        ArrayList<String>  sp = ler("ListaSP.txt");
        ArrayList<String>  words = ler("Palavras.txt");
        

        for (String word : words) {            
            if (sp.contains(word)) {
                indice++;
            } else if (p.contains(word)) {
                indice--;
            }            
        }
        
        if (indice > 0) {
            System.out.println("Positivo");
        } else if (indice < 0) {
            System.out.println("Negativo");
        } else {
            System.out.println("Indefinido");
        }
        
        
    }

    public static ArrayList<String> ler(String file) {
        ArrayList<String> words = new ArrayList();
        
        try {
            
            FileReader in = new FileReader(file);
            BufferedReader reader = new BufferedReader(in);
                        
            while(reader.ready()){
                words.add(reader.readLine());
            }
            
            in.close();
            reader.close();
            
        } catch (IOException e) {
            e.printStackTrace();
        }
        
        
        return words;

    }

}