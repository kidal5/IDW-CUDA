
#include "code/DataManager.h"
#include "code/Constants.h"
#include "code/CpuIdw.h"
#include "code/CpuIdwBase.h"
#include "code/CpuIdwThreaded.h"
#include "code/FullscreenHelper.h"
#include "code/GpuIdwGlobalMemory.cuh"
#include "code/GpuIdwTexture.cuh"
#include "code/HelpPrint.h"
#include "code/Utils.h"


std::vector<CpuIdwBase*> idws;


bool cpuKernelsEnabled = false;
DataManager data;

void refreshIdws(){
	idws.clear();

	if (cpuKernelsEnabled) {
		idws.push_back(new CpuIdw(IMAGE_WIDTH, IMAGE_HEIGHT));
		idws.push_back(new CpuIdwThreaded(IMAGE_WIDTH, IMAGE_HEIGHT));
	}
	
	idws.push_back(new GpuIdwGlobalMemory(IMAGE_WIDTH, IMAGE_HEIGHT));
	idws.push_back(new GpuIdwTexture(IMAGE_WIDTH, IMAGE_HEIGHT, false));
	//this must be initialized after glut has been initialized ... 
	idws.push_back(new GpuIdwTexture(IMAGE_WIDTH, IMAGE_HEIGHT, true));

	data.setNumberOfIdws(idws.size());
};

void drawImage() {
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glClear(GL_COLOR_BUFFER_BIT);
	idws[data.getCurrentIdw()]->drawOpengl(data);
	glutSwapBuffers();
}

void idleFunc() {
	idws[data.getCurrentIdw()]->refresh(data, true);
	data.setChangeDone();

	glutPostRedisplay();
}

static void handleKeys(const unsigned char key, const int x, const int y) {
	if (FullscreenHelper::handleKeys(key, x, y)) return;
	if (HelpPrint::handleKeys(key, x, y)) return;
	if (key == 'v') {
		Utils::toggleVsync();
		return;
	}

	if (key == 'g') {
		cpuKernelsEnabled = !cpuKernelsEnabled;
		refreshIdws();
		return;
	}
	
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


	HelpPrint::print();
    Utils::setVSync(0);

	refreshIdws();

	try {
		glutMainLoop();
	} catch (const std::exception& e) {
		fmt::print(fg(fmt::color::red), "Global exception handler!\n");
		fmt::print(fg(fmt::color::red), "{}", e.what());
	}

    return 0;
}
