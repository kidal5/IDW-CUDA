#include <GL/glut.h>
#include <gl/GL.h>

#include "P2.h"
#include "IdwCpu.h"
#include "AnchorPointsManager.h"

// output image dimensions
const int IMAGE_WIDTH = 768;
const int IMAGE_HEIGHT = 768;

AnchorPointsManager anchor;



std::unique_ptr<IdwBase> idwCpu = std::make_unique<IdwCpu>(IMAGE_WIDTH, IMAGE_HEIGHT);



//zero zero for opengl is left/bottom
//it goes line by line
void drawImage() {
    glClearColor(0.0, 0.0, 0.0, 0.0);
    glClear(GL_COLOR_BUFFER_BIT);
	glDrawPixels(idwCpu->getWidth(), idwCpu->getHeight(), GL_RGB, GL_UNSIGNED_BYTE, idwCpu->getBitmapCpu());

    glutSwapBuffers();
}

void idleFunc() {
    idwCpu->refresh(anchor);
    anchor.setChangeDone();
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


static void handleMouse(const int button, const int state, const int x, const int y) {
    anchor.handleMouse(button, state,x, IMAGE_HEIGHT - y);
}


int main(int argc, char** argv) {

    glutInit(&argc, argv);

    glutInitWindowSize(IMAGE_WIDTH, IMAGE_HEIGHT);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);

    glutCreateWindow("idw");
    glutDisplayFunc(drawImage);
    glutKeyboardFunc(handleKeys);
    glutMouseFunc(handleMouse);

    glutIdleFunc(idleFunc);

    glutMainLoop();

    return 0;
}
