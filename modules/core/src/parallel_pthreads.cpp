/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"

#if defined HAVE_PTHREADS && HAVE_PTHREADS

#include <algorithm>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <stdlib.h>
#include <iostream>
#include <algorithm>
#include <cerrno>

namespace cv
{

class ThreadManager;

enum ForThreadState
{
    eFTNotStarted = 0,
    eFTStarted = 1,
    eFTToStop = 2,
    eFTStoped = 3
};

enum ThreadManagerPoolState
{
    eTMNotInited = 0,
    eTMFailedToInit = 1,
    eTMInited = 2,
    eTMSingleThreaded = 3
};

struct work_load
{
    work_load()
    {
        //std::cout << "workload constructor" << std::endl;
        clear();
    }

    work_load(const cv::Range& range, const cv::ParallelLoopBody& body, int nstripes)
        : m_body(&body), m_range(&range), m_nstripes(nstripes)
    {
        //std::cout << "work_load start " << m_range->start << " end " << m_range->end << " nstripes " << nstripes << std::endl;
        m_blocks_count = ((m_range->end - m_range->start - 1)/m_nstripes) + 1;
        //std::cout << "m_blocks_count " << m_blocks_count << std::endl;
    }

    const cv::ParallelLoopBody* m_body;
    const cv::Range*            m_range;
    int                         m_nstripes;
    unsigned int                m_blocks_count;

    void clear()
    {
        m_body = 0;
        m_range = 0;
        m_nstripes = 0;
        m_blocks_count = 0;
    }
};

class ForThread
{
public:

    ForThread(): m_parent(0), m_state(eFTNotStarted), m_id(0)
    {
        //std::cout << "for thread constructor" << std::endl;
    }

    bool init(size_t id, ThreadManager* parent);

    void run();

    void stop();

    ~ForThread();

private:

    static void* thread_loop_wrapper(void* thread_object);

    void execute();

    void thread_body();

    pthread_t       m_posix_thread;
    pthread_mutex_t m_thread_mutex;
    pthread_cond_t  m_cond_thread_ready;
    pthread_cond_t  m_cond_thread_task;

    ThreadManager*  m_parent;
    ForThreadState  m_state;
    size_t          m_id;

    cv::Range       m_range_to_execute;

    work_load       m_load;
};

class ThreadManager
{
public:
    friend class ForThread;

    static ThreadManager& instance()
    {
        if(!m_instance.ptr)
        {
            pthread_mutex_lock(&m_manager_access_mutex);

            if(!m_instance.ptr)
            {
                m_instance.ptr = new ThreadManager();
            }

            pthread_mutex_unlock(&m_manager_access_mutex);
        }

        return *m_instance.ptr;
    }


    static void stop()
    {
        ThreadManager& manager = instance();

        if(manager.m_pool_state == eTMInited)
        {
            for(size_t i = 0; i < manager.m_num_threads; ++i)
            {
                manager.m_threads[i].stop();
            }
        }

        manager.m_pool_state = eTMNotInited;
    }

    void run(const cv::Range& range, const cv::ParallelLoopBody& body, double nstripes);

    size_t getNumOfThreads();

    void setNumOfThreads(size_t n);

private:

    struct ptr_holder
    {
        ThreadManager* ptr;

        ptr_holder(): ptr(NULL) { }

        ~ptr_holder()
        {
            if(ptr)
            {
                delete ptr;
            }
        }
    };

    ThreadManager();

    ~ThreadManager();

    void wait_complete();

    void notify_complete();

    void initPool();

    size_t defaultNumberOfThreads();

    bool m_is_inited;
    std::vector<ForThread> m_threads;
    size_t m_num_threads;

    pthread_mutex_t m_manager_task_mutex;
    pthread_cond_t  m_cond_thread_task_complete;
    pthread_cond_t  m_cond_complete_recieved;

    unsigned int m_task_position;
    unsigned int m_num_of_completed_tasks;

    static pthread_mutex_t m_manager_access_mutex;
    static ptr_holder m_instance;

    static const char m_env_name[];
    static const size_t m_default_number_of_threads;

    work_load m_work_load;

    struct work_thread_t
    {
        work_thread_t(): value(false) { }
        bool value;
    };

    cv::TLSData<work_thread_t> m_is_work_thread;

    ThreadManagerPoolState m_pool_state;
};
 
pthread_mutex_t ThreadManager::m_manager_access_mutex = PTHREAD_RECURSIVE_MUTEX_INITIALIZER_NP;
ThreadManager::ptr_holder ThreadManager::m_instance;
const char ThreadManager::m_env_name[] = "OPENCV_FOR_THREADS_NUM";
const size_t ThreadManager::m_default_number_of_threads = 8;

ForThread::~ForThread()
{
    //std::cout << "~ForThread" << std::endl;
    if(m_state == eFTStarted)
    {
        stop();

        pthread_mutex_destroy(&m_thread_mutex);

        pthread_cond_destroy(&m_cond_thread_ready);

        pthread_cond_destroy(&m_cond_thread_task);
    }
}

bool ForThread::init(size_t id, ThreadManager* parent)
{
    m_id = id;

    m_parent = parent;

    int res = 0;

    res |= pthread_mutex_init(&m_thread_mutex, NULL);

    res |= pthread_cond_init(&m_cond_thread_ready, NULL);

    res |= pthread_cond_init(&m_cond_thread_task, NULL);

    if(!res)
    {
        res = pthread_mutex_lock(&m_thread_mutex);

        if(!res)
        {
            res = pthread_create(&m_posix_thread, NULL, thread_loop_wrapper, (void*)this);

            if(!res)
            {
                pthread_cond_wait(&m_cond_thread_ready, &m_thread_mutex);

                pthread_mutex_unlock(&m_thread_mutex);
            }
        }
    }


    return res == 0;
}

void ForThread::stop()
{
    //std::cout << "stoping... " << std::endl;
    if(m_state == eFTStarted)
    {
        m_state = eFTToStop;

        run();

        pthread_join(m_posix_thread, NULL);

        //std::cout << "thread id " << m_id << " joined " << std::endl;
    }

    m_state = eFTStoped;
}

void ForThread::run()
{
    pthread_mutex_lock(&m_thread_mutex);

    pthread_cond_signal(&m_cond_thread_task);

    pthread_mutex_unlock(&m_thread_mutex);
}

void* ForThread::thread_loop_wrapper(void* thread_object)
{
    ((ForThread*)thread_object)->thread_body();
    return 0;
}

void ForThread::execute()
{
    size_t m_current_pos = CV_XADD(&m_parent->m_task_position, 1);

    //std::cout << "taking portion " << m_current_pos << std::endl;

    work_load& load = m_parent->m_work_load;

    while(m_current_pos < load.m_blocks_count)
    {
        int start = load.m_range->start + m_current_pos*load.m_nstripes;
        int end = std::min(start + load.m_nstripes, load.m_range->end);

        //std::cout << "start " << start << " end " << end << " " << load.m_nstripes << std::endl;

        m_parent->m_work_load.m_body->operator()(cv::Range(start, end));

        m_current_pos = CV_XADD(&m_parent->m_task_position, 1);
    }
}

void ForThread::thread_body()
{
    //std::cout << "thread body starts" << std::endl;

    m_parent->m_is_work_thread.get()->value = true;

    pthread_mutex_lock(&m_thread_mutex);

    pthread_cond_signal(&m_cond_thread_ready);

    m_state = eFTStarted;

    while(m_state == eFTStarted)
    {
        //std::cout << "go to sleep " << m_id << std::endl;
        pthread_cond_wait(&m_cond_thread_task, &m_thread_mutex);

        if(m_state == eFTStarted)
        {
            //std::cout << "wakeup signal recieved " << m_id << std::endl;
            execute();

            m_parent->notify_complete();
        }
    }

    pthread_mutex_unlock(&m_thread_mutex);
}

ThreadManager::ThreadManager(): m_num_threads(0), m_num_of_completed_tasks(0), m_pool_state(eTMNotInited)
{
    int res = 0;

    res |= pthread_mutex_init(&m_manager_task_mutex, NULL);

    res |= pthread_cond_init(&m_cond_thread_task_complete, NULL);

    res |= pthread_cond_init(&m_cond_complete_recieved, NULL);

    if(!res)
    {
        //std::cout << "Init ok" << std::endl;

        setNumOfThreads(defaultNumberOfThreads());

        m_task_position = 0;
    }
    else
    {
        //std::cout << "Failed to init " << m_num_threads << std::endl;

        m_num_threads = 1;
        m_pool_state = eTMFailedToInit;
        m_task_position = 0;

        //print error;
    }
}

ThreadManager::~ThreadManager()
{
    stop();

    pthread_mutex_destroy(&m_manager_task_mutex);

    pthread_cond_destroy(&m_cond_thread_task_complete);

    pthread_cond_destroy(&m_cond_complete_recieved);

    pthread_mutex_destroy(&m_manager_access_mutex); 
}

void ThreadManager::run(const cv::Range& range, const cv::ParallelLoopBody& body, double nstripes)
{
    int res = -1;

    bool is_work_thread;

    is_work_thread = m_is_work_thread.get()->value;

    if(!is_work_thread)
    {
        res = pthread_mutex_trylock(&m_manager_access_mutex);
    }

    if(!res)
    {
        if(getNumOfThreads() > 1)
        {
            //std::cout << "running in multi thread mode " << std::endl;

            initPool();

            m_task_position = 0;

            if(range.end - range.start > 1)
            {
                //std::cout << "pool inited " << std::endl;

                //if(nstripes <= 0)
                {
                    //std::cout << " range.start " << range.start << " range.end " << range.end << " m_threads.size()" << m_threads.size() << std::endl;
                    //nstripes = double(range.end - range.start)/(2*m_threads.size());
                    nstripes = double(range.end - range.start)/(4*m_threads.size());
                    //std::cout << " range.start " << range.start << " range.end " << range.end << " m_threads.size()" << m_threads.size() << std::endl;
                }

                //std::cout << "nstripes " << nstripes << std::endl;

                pthread_mutex_lock(&m_manager_task_mutex);
                
                m_num_of_completed_tasks = 0;

                m_work_load = work_load(range, body, std::ceil(nstripes));

                for(size_t i = 0; i < m_threads.size(); ++i)
                {
                    m_threads[i].run();
                }

                wait_complete();
            }
            else
            {
                pthread_mutex_unlock(&m_manager_access_mutex);

                body(range);
            }
        }
        else
        {
            pthread_mutex_unlock(&m_manager_access_mutex);

            //std::cout << "running in single thread mode " << std::endl;
            
            body(range);
        }
    }
    else
    {
        //std::cout << "running in single thread mode " << std::endl;

        body(range);
    }
}

void ThreadManager::wait_complete()
{

    //std::cout << "starting wait for complition" << std::endl;
    
    pthread_cond_wait(&m_cond_thread_task_complete, &m_manager_task_mutex);

    pthread_mutex_unlock(&m_manager_task_mutex);

    pthread_mutex_unlock(&m_manager_access_mutex);
}

void ThreadManager::notify_complete()
{

    unsigned int comp = CV_XADD(&m_num_of_completed_tasks, 1);

    if(comp == (m_num_threads - 1))
    {
        pthread_mutex_lock(&m_manager_task_mutex);

        pthread_cond_signal(&m_cond_thread_task_complete);

        pthread_mutex_unlock(&m_manager_task_mutex);
    }
}

void ThreadManager::initPool()
{
    if(m_pool_state != eTMNotInited || m_num_threads == 1)
        return;

    //std::cout << "resizing threads " <<  m_num_threads << std::endl;
    m_threads.resize(m_num_threads);

    //std::cout << "finish resizing" << std::endl;
    for(size_t i = 0; i < m_threads.size(); ++i)
    {
        m_threads[i].init(i, this);
    }

    m_pool_state = eTMInited;
}

size_t ThreadManager::getNumOfThreads()
{
    return m_num_threads;
}

void ThreadManager::setNumOfThreads(size_t n)
{
    int res = pthread_mutex_lock(&m_manager_access_mutex);

    if(!res)
    {
        if(n == 0)
        {
            n = defaultNumberOfThreads();
        }

        if(n != m_num_threads && m_pool_state != eTMFailedToInit)
        {
            if(m_pool_state == eTMInited)
            {
                stop();
                m_threads.clear();
            }

            m_num_threads = n;

            if(m_num_threads == 1)
            {
                m_pool_state = eTMSingleThreaded;
            }
            else
            {
                m_pool_state = eTMNotInited;
            }
        }

        pthread_mutex_unlock(&m_manager_access_mutex);
    }
}

size_t ThreadManager::defaultNumberOfThreads()
{
    size_t result;

    char * env = getenv(m_env_name);

    if(!env || (env && sscanf(env, "%lu", &result) != 1))
    {
        result = m_default_number_of_threads;
    }

    return result;
}

void parallel_for_pthreads(const cv::Range& range, const cv::ParallelLoopBody& body, double nstripes)
{
    ThreadManager::instance().run(range, body, nstripes);
}

}

#endif
