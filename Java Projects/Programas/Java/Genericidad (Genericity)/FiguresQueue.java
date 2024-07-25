package practica2;

import practica1.Figure;
import practica2.librerias.implementaciones.QueueAL;

/**
 * class FiguresQueue.
 * 
 * @author LTP 
 * @version 2020-21
 */

public class FiguresQueue<T extends Figure> extends QueueAL<T> { 
    public double area(){
        double suma =0;
        for(int i=0;i<this.size();i++){
            suma = suma + this.area();
        }
        return suma;
    }
}