package practica1;


/**
 * Write a description of class Square here.
 *
 * @author (your name)
 * @version (a version number or a date)
 */
public class Square extends Rectangle {
    private double lado;
    
    public Square(double x, double y, double l){
        super(x,y,l,l);
        lado = l;
        
    }
}
