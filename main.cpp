
#include "code/DataManager.h"
#include "code/CpuIdw.h"
#include "code/CpuIdwBase.h"
#include "code/CpuIdwThreaded.h"
#include "code/FullscreenHelper.h"
#include "code/GpuIdwGlobalMemory.cuh"
#include "code/GpuIdwTexture.cuh"
#include "code/HelpPrint.h"
#include "code/Utils.h"


int currentWidth = 768;
int currentHeight= 768;

std::vector<CpuIdwBase*> idws;


bool cpuKernelsEnabled = false;
DataManager data;

void refreshIdws(){
	for (int i = idws.size() - 1; i >= 0; --i) 
		delete idws[i];
	idws.clear();

	if (cpuKernelsEnabled) {
		idws.push_back(new CpuIdw(currentWidth, currentHeight));
		idws.push_back(new CpuIdwThreaded(currentWidth, currentHeight));
	}
	
	idws.push_back(new GpuIdwGlobalMemory(currentWidth, currentHeight));
	idws.push_back(new GpuIdwTexture(currentWidth, currentHeight, false));
	
	//this must be initialized after glut has been initialized ... 
	idws.push_back(new GpuIdwTexture(currentWidth, currentHeight, true));

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

void reshape(int w, int h) {
	fmt::print("{} {}\n", w,h);

	
	if (w < 256 || h < 256) {
		glutReshapeWindow(256, 256);
		return;
	}

	if (w % 2 == 1 && h % 2 == 1) {
		glutReshapeWindow(w+1, h + 1);
	} else if (w % 2 == 1) {
		glutReshapeWindow(w + 1,h);
	} else if (h % 2 == 1) {
		glutReshapeWindow(w, h + 1);
	} else {
		currentWidth = w;
		currentHeight = h;
		refreshIdws();
	}

	
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
	data.handleMouse(button, state, x, currentHeight - y);
}


int main(int argc, char** argv) {
    
    glutInit(&argc, argv);

    glutInitWindowSize(currentWidth, currentHeight);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);

    glutCreateWindow("idw");
    glutDisplayFunc(drawImage);
    glutKeyboardFunc(handleKeys);
    glutSpecialFunc(handleSpecialKeys);
    glutMouseFunc(handleMouse);
    glutIdleFunc(idleFunc);
	//glutReshapeFunc(reshape);


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
