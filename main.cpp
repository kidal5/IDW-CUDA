#include <GL/glut.h>
#include <gl/GL.h>

#include "IdwCpu.h"

// output image dimensions
const int IMAGE_WIDTH = 768;
const int IMAGE_HEIGHT = 768;

IdwCpu idwCpu = IdwCpu(IMAGE_WIDTH, IMAGE_HEIGHT);


void drawImage() {
    glClearColor(0.0, 0.0, 0.0, 0.0);
    glClear(GL_COLOR_BUFFER_BIT);
	glDrawPixels(idwCpu.width, idwCpu.height, GL_RGB, GL_UNSIGNED_BYTE, idwCpu.bitmap.get());
    glutSwapBuffers();
}

void idleFunc() {
    idwCpu.refresh();
    glutPostRedisplay();
}

static void handleKeys(const unsigned char key, int x, int y) {
    switch (key) {
    case 27:	// ESC
        exit(0);
    default: 
        break;
    }
}


int main(int argc, char** argv) {

    glutInit(&argc, argv);

    glutInitWindowSize(IMAGE_WIDTH, IMAGE_HEIGHT);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);

    glutCreateWindow("idw");
    glutDisplayFunc(drawImage);
    glutKeyboardFunc(handleKeys);

    glutIdleFunc(idleFunc);

    glutMainLoop();

    return 0;
}
