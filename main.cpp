#include <GL/glut.h>
#include <gl/GL.h>

#include "P2.h"
#include "IdwCpu.h"
#include "IdwThreaded.h"
#include "AnchorPointsManager.h"
#include "Utils.h"

// output image dimensions
const int IMAGE_WIDTH = 768;
const int IMAGE_HEIGHT = 768;
const P2 imgSize = P2(IMAGE_WIDTH, IMAGE_HEIGHT);

AnchorPointsManager anchor;



std::unique_ptr<IdwBase> idw = std::make_unique<IdwThreaded>(IMAGE_WIDTH, IMAGE_HEIGHT);



//zero zero for opengl is left/bottom
//it goes line by line
void drawImage() {
    glClearColor(0.0, 0.0, 0.0, 0.0);
    glClear(GL_COLOR_BUFFER_BIT);

	glDrawPixels(idw->getWidth(), idw->getHeight(), GL_RGB, GL_UNSIGNED_BYTE, idw->getBitmapCpu());
	
    glutSwapBuffers();
}

void idleFunc() {
    idw->refresh(anchor);
    Utils::drawGui(idw->getFps(), idw->getMethodName(), anchor.getMouseValue(), anchor.getPParam(), idw->getBitmapCpu(), imgSize);
    anchor.setChangeDone();

	glutPostRedisplay();
}

static void handleKeys(const unsigned char key, const int x, const int y) {
    anchor.handleKeys(key, x, y);
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
