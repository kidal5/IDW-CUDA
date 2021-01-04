#include "FullscreenHelper.h"

#include <GL/glut.h>

int FullscreenHelper::isFullScreen = 0;
int FullscreenHelper::previousPosition[2] = { 0, 0 };
int FullscreenHelper::previousSize[2] = { 100, 100 };


bool FullscreenHelper::handleKeys(const unsigned char key, const int x, const int y) {

    if (key != 'f')
        return false;

	
    if (isFullScreen) {
        setWindowed(previousPosition[0], previousPosition[1], previousSize[0], previousSize[1]);
    }
    else {
        previousPosition[0] = glutGet(GLUT_WINDOW_X);
        previousPosition[1] = glutGet(GLUT_WINDOW_Y);
        previousSize[0] = glutGet(GLUT_WINDOW_WIDTH);
        previousSize[1] = glutGet(GLUT_WINDOW_HEIGHT);
        glutFullScreen();
    }
    isFullScreen = !isFullScreen;

    return true;
}

void FullscreenHelper::setWindowed(const int positionX, const int positionY, const int width, const int height) {
    glutReshapeWindow(width, height);
    glutPositionWindow(positionX, positionY);
}