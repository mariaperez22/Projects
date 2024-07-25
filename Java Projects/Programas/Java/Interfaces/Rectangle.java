package practica3;
import java.util.*;

/**
 * Write a description of class Rectangle here.
 *
 * @author (your name)
 * @version (a version number or a date)
 */
public class Rectangle extends Figure implements ComparableRange<Figure> {
    private double base;
    private double height;
    
    public double area(){
        return base*height;
    }

    public Rectangle(double x, double y, double b, double h) {
        super(x, y);
        base = b;
        height = h;
    }

    public String toString() {
        return "Rectangle:\n\t" +
            super.toString() +
            "\n\tBase: " + base +
            "\n\tHeight: " + height;
    }
    
    public boolean equals(Object o){
        if (!(o instanceof Rectangle)) { return false; }
        Rectangle r = (Rectangle) o;
        return this.base == r.base && this.height == r.height && super.equals(r);
        
    }
    
    public int compareToRange(Figure o){
        if ( Math.abs(this.area() - o.area()) > 0.1*(o.area() + this.area())){
            return -1;
        } else{
            return 0;
        }
    }
    
    public void print (char c){
        int b= (int) base;
        int h= (int) height;
        for(int i=0;i<h;i++){
            for(int j=0;j<b;j++){
                System.out.print(c);
            }
            System.out.println();
        }
    }
}
