#include<iostream>
#include<fstream>
using namespace std;

bool isEndOfSentence (char c)
{
	if ('a' <= c && c <= 'z')
		return true;
	if ('A' <= c && c <= 'Z')
		return true;
	if ('0' <= c && c <= '9')
		return true;

	return false;
}

int main()
{
	char prev = ' ';
	char cur;
	char next;
	bool checkIfSpace = false;

	ifstream fin ("wiki.txt");
	ofstream fout ("wiki_conv.txt");

	fin>>cur;

	while (fin >> noskipws >> next)
	{
		if (cur == '.' && isEndOfSentence(prev) && !isEndOfSentence(next))
		{
			fout << cur << '\n';
			checkIfSpace = true;
		}
		else
		{
			if (!(checkIfSpace && cur == ' '))
				fout <<cur;
			checkIfSpace = false;
		}

		prev = cur;
		cur = next;
	}

	fout << next << '\n';

	fin.close();
	fout.close();

	return 0;
}