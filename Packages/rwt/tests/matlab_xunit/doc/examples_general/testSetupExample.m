function test_suite = testSetupExample
initTestSuite;

function fh = setup
fh = figure;

function teardown(fh)
delete(fh);

function testColormapColumns(fh)
assertEqual(size(get(fh, 'Colormap'), 2), 3);

function testPointer(fh)
assertEqual(get(fh, 'Pointer'), 'arrow');
