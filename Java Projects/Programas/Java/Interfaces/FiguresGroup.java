package practica3;
import java.util.*;


/**
 * class FiguresGroup.
 * 
 * @author LTP 
 * @version 2020-21
 */

public class FiguresGroup {
    private static final int NUM_FIGURES = 10;
    private Figure[] figuresList = new Figure[NUM_FIGURES];
    private int numF = 0;
    
    public void add(Figure f) { figuresList[numF++] = f; }
    
    public String toString() {
        String s = "";
        for (int i = 0; i < numF; i++) {
            s += "\n" + figuresList[i];
        }
        return s;
    }

    public boolean found(Figure f) {
        for (int i = 0; i < numF; i++) {
        	if (figuresList[i].equals(f)) return true;
        }
        return false;
    }

    private boolean included(FiguresGroup g) {
        for (int i = 0; i < g.numF; i++) {
        	if (!found(g.figuresList[i])) return false;
        }
		return true;
    }
    
    public boolean equals(Object o){
        if (!(o instanceof FiguresGroup)) { return false; }
        FiguresGroup fg = (FiguresGroup) o;
        return this.included(fg) && fg.included(this);
        
    }
    
    public double areaTotal(){
        double suma = 0;
        for(int i = 0;i < numF; i++){
            suma = suma + figuresList[i].area();
        }
        return suma;
    }
    
    public double greatestFigure(){
        double mayor = figuresList[0].area();
        for (int i=0;i<numF; i++){
            if(figuresList[i].area()>mayor){
                mayor = figuresList[i].area();
            }
        }
        return mayor;
    }
    
    public List<Figure> orderedList(){
        List<Figure> l = new ArrayList<Figure>();
        for (int i=0;i<numF;i++){
            Figure x = figuresList[i];
            int j=i-1;
            while(j>=0 && l.get(j).CompareTo(x) > 0){
                j--;
            }
            l.add(j+1,x);
        }
         return l;
    }
    
    public void print(char c){
        for (int i=0;i<numF;i++){
            if(figuresList[i] instanceof Printable){
                Printable v = (Printable) figuresList[i];
                v.print(c);
            }
        }
    }
}