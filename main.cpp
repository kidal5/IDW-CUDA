//#include <gl/GL.h>

#include "DataManager.h"
#include "Utils.h"
#include "Constants.h"
#include "CpuIdw.h"
#include "CpuIdwBase.h"
#include "CpuIdwThreaded.h"
#include "GpuIdwGlobalMemory.cuh"
#include "GpuIdwTexture.cuh"


DataManager data;

std::unique_ptr<CpuIdwBase> idws[4] = {
	std::make_unique<CpuIdw>(IMAGE_WIDTH, IMAGE_HEIGHT),
	std::make_unique<CpuIdwThreaded>(IMAGE_WIDTH, IMAGE_HEIGHT),
	std::make_unique<GpuIdwGlobalMemory>(IMAGE_WIDTH, IMAGE_HEIGHT),
	std::make_unique<GpuIdwTexture>(IMAGE_WIDTH, IMAGE_HEIGHT),
};

void drawImage() {
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glClear(GL_COLOR_BUFFER_BIT);

	auto& idw = idws[data.getCurrentIdw()];
	glDrawPixels(idw->getWidth(), idw->getHeight(), GL_LUMINANCE, GL_UNSIGNED_BYTE, idw->getBitmapGreyscaleCpu());

	glutSwapBuffers();
}

void idleFunc() {
	auto& idw = idws[data.getCurrentIdw()];
	idw->refresh(data, true);
	Utils::drawGui(idw->getFps(), idw->getMethodName(), data.getMouseValue(), data.getPParam(), idw->getBitmapGreyscaleCpu(), imgSize, 150);
	data.setChangeDone();

	glutPostRedisplay();
}

static void handleKeys(const unsigned char key, const int x, const int y) {
	data.handleKeys(key, x, y);
}

static void handleSpecialKeys(const int key, const int x, const int y) {
	data.handleSpecialKeys(key, x, y);
}

static void handleMouse(const int button, const int state, const int x, const int y) {
	data.handleMouse(button, state, x, IMAGE_HEIGHT - y);
}

int main(int argc, char** argv) {

	glutInit(&argc, argv);

	glutInitWindowSize(IMAGE_WIDTH, IMAGE_HEIGHT);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);

	glutCreateWindow("idw");
	glutDisplayFunc(drawImage);
	glutKeyboardFunc(handleKeys);
	glutSpecialFunc(handleSpecialKeys);
	glutMouseFunc(handleMouse);

	glutIdleFunc(idleFunc);

	glutMainLoop();

	return 0;
}
